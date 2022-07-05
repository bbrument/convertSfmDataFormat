import argparse
import numpy as np
import os
import json

def read_json_data(path):

    with open(path, 'r') as f:
        data = json.load(f)

    cameras = data['poses']
    images = data['views']
    points3D = data['points']
    intrinsics = data['intrinsics']
    
    return cameras, images, points3D, intrinsics

def load_json_data(path):

    # Json data
    cameras, images, points3D, intrinsics = read_json_data(path)
    
    # Intrinsics data
    w, h, f = intrinsics['width'], intrinsics['height'], intrinsics['pxFocalLength'][0]
    ppx, ppy = intrinsics['principalPoint'][0], intrinsics['principalPoint'][1]
    px = w/2 + ppx
    py = h/2 + ppy
    intrinsics_list = [w,h,f,px,py]

    # Image paths
    images_path = [f['path'] for f in images]

    # Camera poses
    quaternion_list = []
    translation_list = []
    for cam in cameras:

        # Get world2cam poses
        R = np.transpose(np.array(cam['pose']['transform']['rotation']).reshape([3,3])) # R (world2cam)
        cam_center = np.expand_dims(np.array(cam['pose']['transform']['center']), axis=1) # -R^T * t (cam2world)
        t = -R @ cam_center # t (world2cam)
        translation_list.append(np.squeeze(np.transpose(t)))

        # Convert in rotation matrix in quaternion
        qw = np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])/2)
        qx = (R[2,1] - R[1,2])/(4*qw)
        qy = (R[0,2] - R[2,0])/(4*qw)
        qz = (R[1,0] - R[0,1])/(4*qw)
        quaternion_list.append([qw,qx,qy,qz])

    # Convert to numpy arrays
    translation_list = np.array(translation_list)
    quaternion_list = np.array(quaternion_list)

    print("JSON data read from {}".format(path))
    return intrinsics_list, images_path, quaternion_list, translation_list


def gen_poses(basedir, jsonpath):

    print('Reading JSON data from {}'.format(jsonpath))
    intrinsics_list, images_path, quaternion_list, translation_list = load_json_data(jsonpath)
    
    # Save cameras.txt
    camera_model = "SIMPLE_PINHOLE"
    intr_line = "{} {} {} {} {} {} {}\n".format(1, camera_model, intrinsics_list[0], 
        intrinsics_list[1], intrinsics_list[2], intrinsics_list[3], intrinsics_list[4])

    intro_cam_lines = ['# Camera list with one line of data per camera:\n',
                        '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n',
                        '# Number of cameras: 1\n']
    with open(os.path.join(basedir, "cameras.txt"), "w+") as f:
        f.writelines(intro_cam_lines)
        f.writelines(intr_line)

    # Save quaternions, translations and images paths
    intro_img_lines = ['# Image list with two lines of data per image:\n',
                        '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n',
                        '#   POINTS2D[] as (X, Y, POINT3D_ID)\n']

    images_lines = []      
    for i in range(len(images_path)):

        quater = quaternion_list[i,:]
        t = translation_list[i,:]

        image_name = images_path[i].split('/')[-1]

        line = "{} {} {} {} {} {} {} {} {} {}\n".format(i+1, quater[0], quater[1], quater[2], quater[3], t[0], t[1], t[2], 1, image_name)
        images_lines.append(line)
        images_lines.append("\n")
    
    with open(os.path.join(basedir, "images.txt"), "w+") as f:
        f.writelines(intro_img_lines)
        f.writelines(images_lines)

    # Save empty points3D.txt
    intro_pts3d_lines = ['# 3D point list with one line of data per point:\n',
        '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n']
    with open(os.path.join(basedir, "points3D.txt"), "w+") as f:
            f.writelines(intro_pts3d_lines)
    
    return True


# Example usage:
# python process_data.py
parser = argparse.ArgumentParser()
parser.add_argument('--scenedir', type=str, default='data',
                    help='Data directory. e.g. data')
parser.add_argument('--jsonfile', type=str, default='sfm.json', 
                    help='The name of sfm json')
args = parser.parse_args()


if __name__=='__main__':
    gen_poses(args.scenedir, args.jsonfile)
