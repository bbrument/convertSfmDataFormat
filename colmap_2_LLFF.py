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

    if not os.path.exists(os.path.join(basedir, "images")):
        os.makedirs(os.path.join(basedir, "images"))

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
        shutil.copy(imagePaths[i], os.path.join(basedir,'images/'))

    if downsample_factor != 1.0:
        minify(basedir, 'images/', 'images_'+str(downsample_factor), downsample_factor)

    # save poses_bounds.npy
    pts_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)        # Append all point cloud points to list
    pts_arr = np.array(pts_arr)             # shape=(len(pts3d), 3)
    
    valid_z = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    print(valid_z.shape)
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in range(valid_z.shape[-1]):
        zs = valid_z[:, i]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    
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
        os.mkdir(path_out)

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
