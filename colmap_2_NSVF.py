import argparse
import numpy as np
import cv2
import os
import shutil

import colmap_read_model as read_model

def load_colmap_data(realdir, foldername="sparse/0", from_cam_files=False):
    """
    Returns:
        images: np array (h, w, 3, num_images)
        poses: np array (3, 5, num_images), [R t] (does not involve intrinsic matrix).
        bds: np array (2, num_images)
    """

    camerasfile = os.path.join(realdir, '{}/cameras.bin'.format(foldername))
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # We just take the first camera settings
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print(cam)

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    imagesfile = os.path.join(realdir, '{}/images.bin'.format(foldername))
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]

    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, '{}/points3D.bin'.format(foldername))
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    print("COLMAP data read from {}".format(foldername))
    return poses, pts3d, perm

def absoluteFilePaths(directory):
    file_paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return file_paths

def gen_poses(basedir, foldername, downsample_factor):

    if not os.path.exists(os.path.join(basedir, "pose")):
        os.makedirs(os.path.join(basedir, "pose"))
    if not os.path.exists(os.path.join(basedir, "rgb")):
        os.makedirs(os.path.join(basedir, "rgb"))

    sparseFoldername = os.path.join(foldername, "sparse/0/")
    imagesFoldername = os.path.join(foldername, "images/")

    print('Reading COLMAP output from {}'.format(foldername))
    
    poses, pts3d, perm = load_colmap_data(basedir, foldername=sparseFoldername)
    print("poses.shape: {} ([3, 5, num_images])".format(poses.shape))
    print("hwf (orig/resized): {} / {}".format(poses[:,4,0], np.round(poses[:,4,0] / float(downsample_factor))))
    
    print("Average camera center: {}".format(poses[:3, 3, :].mean(-1)))

    # Image paths
    imagePaths = absoluteFilePaths(imagesFoldername)

    # save images
    for i in range(len(imagePaths)):
        shutil.copy(imagePaths[i],os.path.join(basedir,"rgb"))

    if downsample_factor != 1.0:
        minify(basedir, 'rgb/', 'rgb_'+str(downsample_factor), downsample_factor)
    
    # save intrinsics.txt
    hwf = np.round(poses[:,4,0] / float(downsample_factor))
    h,w,f = hwf[0], hwf[1], hwf[2]
    lines = [
        "{} 0.0 {} 0.0\n".format(float(f), float(w/2)),
        "0.0 {} {} 0.0\n".format(float(f), float(h/2)),
        "0.0 0.0 1.0 0.0\n",
        "0.0 0.0 0.0 1.0\n",
    ]
    with open(os.path.join(basedir, "intrinsics.txt"), "w+") as f:
        f.writelines(lines)

    # Save camera extrinsics
    for i in range(poses.shape[-1]):
        pose = poses[:,:,i]
        """
        Coordinate axis transform: 
        NeRF (lego, train 0):
            -0.9999021887779236 0.004192245192825794 -0.013345719315111637 -0.05379832163453102
            -0.013988681137561798 -0.2996590733528137 0.95394366979599 3.845470428466797
            -4.656612873077393e-10 0.9540371894836426 0.29968830943107605 1.2080823183059692
            0.0 0.0 0.0 1.0        
        NSVF (lego, train 0):
            -0.9999021887779236 -0.004192245192825794 0.013345719315111637 -0.05379832163453102
            -0.013988681137561798 0.2996590733528137 -0.95394366979599 3.845470428466797
            -4.656612873077393e-10 -0.9540371894836426 -0.29968830943107605 1.2080823183059692
            0.0 0.0 0.0 1.0
        """
        lines = [
            "{} {} {} {}\n".format(pose[0,0], -pose[0,1], -pose[0,2], pose[0,3]),
            "{} {} {} {}\n".format(pose[1,0], -pose[1,1], -pose[1,2], pose[1,3]),
            "{} {} {} {}\n".format(pose[2,0], -pose[2,1], -pose[2,2], pose[2,3]),
            "0.0 0.0 0.0 1.0\n",
        ]
        with open(os.path.join(basedir, "pose", "%08d_pose.txt" % i), "w+") as f:
            f.writelines(lines)
            

    # save bbox.txt
    pts_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)        # Append all point cloud points to list
    pts_arr = np.array(pts_arr)             # shape=(len(pts3d), 3)
    min = np.array([np.percentile(pts_arr[:,i], 2) for i in range(3)]) #/ float(downsample_factor)
    max = np.array([np.percentile(pts_arr[:,i], 98) for i in range(3)]) #/ float(downsample_factor)
    size = max-min
    min -= size*0.1     # 10% margin beyond max/min (percentile) for safety
    max += size*0.1
    voxel_size = np.mean(size) / 8.        # Initialize voxel sizes to 1/8 of volume (approx. 8^3 voxels)
    bbox_line = "{} {} {} {} {} {} {}".format(min[0], min[1], min[2], max[0], max[1], max[2], voxel_size)
    with open(os.path.join(basedir, "bbox.txt"), "w+") as f:
        f.write(bbox_line)
    
    return True


def minify(basedir, indir, outdir, downsample_factor):
    """
    Folder structure:
    basedir
    |---indir
    |---outdir
    """
    from subprocess import check_output
    
    path_in = os.path.join(basedir, indir)
    path_out = os.path.join(basedir, outdir)
    
    imgs = [os.path.join(path_in, f) for f in sorted(os.listdir(path_in))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    
    resizearg = '{}%'.format(int(100./downsample_factor))

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    check_output('cp {}/* {}'.format(path_in, path_out), shell=True)
    if downsample_factor == 1:
        print("Factor=1, original resolution images copied to {}".format(os.path.join(basedir, outdir)))
        return
    
    ext = imgs[0].split('.')[-1]
    args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
    print(args)
    os.chdir(path_out)
    check_output(args, shell=True)
    
    if ext != 'png':
        os.chdir(basedir)
        check_output('rm {}/*.{}'.format(outdir, ext), shell=True)
    print('Finished downsizing images.')


def preprocess_imgs(basedir, indir, outdir, downsample_factor, tomask=False, maskdir=None):
    path_in = os.path.join(basedir, indir)
    path_out = os.path.join(basedir, outdir)
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    files = [os.path.join(path_in, f) for f in sorted(os.listdir(path_in))]
    imgs_fn = [f for f in files if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    if tomask:
        path_mask = os.path.join(basedir, maskdir)
        files = [os.path.join(path_mask, f) for f in sorted(os.listdir(path_mask))]
        masks_fn = [f for f in files if f.endswith('npy')]
        assert len(imgs_fn) == len(masks_fn)

    size = cv2.imread(imgs_fn[0]).shape
    size = (round(size[1] / float(downsample_factor)), round(size[0] / float(downsample_factor)))
    for i in range(len(imgs_fn)):
        img = cv2.resize(cv2.imread(imgs_fn[i], cv2.IMREAD_UNCHANGED), size)

        if tomask:
            mask = cv2.resize((np.load(masks_fn[i]) > 1.E-10).astype(np.uint8)*255, size)
            img[mask==0] = 0
            img = np.append(img, mask[:,:,None], axis=-1)

        cv2.imwrite(
            os.path.join(path_out, os.path.basename(imgs_fn[i])).replace(".JPG", ".png"), 
            img
        )



# Example usage:
# python process_data.py
parser = argparse.ArgumentParser()
parser.add_argument('--scenedir', type=str, default='lego_data_dir',
                    help='Data directory. e.g. lego_data_dir')
parser.add_argument('--colmapdir', type=str, default='sparse/0', 
                    help='The name of folder to load colmap .bin data from. COLMAP defualt is \'sparse/0\'.')
parser.add_argument('--downsample_factor', type=int, default=1)
args = parser.parse_args()


if __name__=='__main__':
    gen_poses(args.scenedir, args.colmapdir, args.downsample_factor)
