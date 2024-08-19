import os
import numpy as np
import json
import sys

def read_view(view, intrinsics, poses):

    # Get image path
    path = view['path']
    
    # Get pose and intrinsic ids
    pose_id = view['poseId']
    intrinsic_id = view['intrinsicId']

    # Get intrinsic and pose data
    intrinsic = intrinsics[intrinsic_id]
    pose = poses[pose_id]
            
    # Get width, height
    width = float(intrinsic['width'])
    height = float(intrinsic['height'])

    # Get focal length
    if 'pxFocalLength' in intrinsic:
        fx = float(intrinsic['pxFocalLength'][0])
        fy = float(intrinsic['pxFocalLength'][1])
    else:
        sensor_width = float(intrinsic['sensorWidth'])
        sensor_height = float(intrinsic['sensorHeight'])
        focal_length = float(intrinsic['focalLength'])
        fx = focal_length * width / sensor_width
        fy = focal_length * height / sensor_height

    # Get principal point
    cx = width / 2 + float(intrinsic['principalPoint'][0])
    cy = height / 2 + float(intrinsic['principalPoint'][1])

    # Get intrinsics matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Get pose data
    R_c2w = np.array(pose['pose']['transform']['rotation'], dtype=np.float32).reshape([3,3])
    center = np.expand_dims(np.array(pose['pose']['transform']['center'], dtype=np.float32), axis=1)

    return K, R_c2w, center, path


def read_cameras_sfm(cameras_sfm_path, undisto_images_path=None):
    with open(cameras_sfm_path, 'r') as f:
        cameras_sfm = json.load(f)
    
    # Get views, intrinsics, poses
    views = cameras_sfm['views']
    intrinsics = cameras_sfm['intrinsics']
    poses = cameras_sfm['poses']

    views_data = {}
    intrinsics_data = {}
    poses_data = {}
    for view in views:
        views_data[view['viewId']] = view
        if undisto_images_path is not None:
            view['undistortedImagePath'] = os.path.join(undisto_images_path, view['viewId'] + '.png')
            if not os.path.exists(view['undistortedImagePath']):
                view['undistortedImagePath'] = os.path.join(undisto_images_path, view['viewId'] + '.exr')
    
    for intrinsic in intrinsics:
        intrinsics_data[intrinsic['intrinsicId']] = intrinsic
    
    for pose in poses:
        poses_data[pose['poseId']] = pose

    return views_data, intrinsics_data, poses_data

def read_meshroom_project(meshroom_project_path, cameras_sfm_path=None):
    with open(meshroom_project_path, 'r') as f:
        data = json.load(f)

    # Get nodes and graph
    cache_folder = os.path.join(os.path.dirname(meshroom_project_path), 'MeshroomCache')
    nodes = data['header']['nodesVersions'] # {'CameraInit': '10.0', 'ConvertSfMFormat': '2.0', 'DepthMapFilter': '4.0', 'Meshing': '7.0', 'Texturing': '6.0', 'MeshFiltering': '3.0', 'FeatureMatching': '2.0', 'DepthMap': '5.0', 'PrepareDenseScene': '3.1', 'FeatureExtraction': '1.3', 'ImageMatching': '2.0', 'StructureFromMotion': '3.3'}
    graph = data['graph']

    # Get 'cameras.sfm' path
    print("Looking for 'cameras.sfm'...")
    if cameras_sfm_path is not None:
        print(f"'cameras.sfm' found in the input path.")
    elif 'StructureFromMotion' in nodes:
        cameras_sfm_path = os.path.join(cache_folder, graph['StructureFromMotion_1']['nodeType'], 
                                        graph['StructureFromMotion_1']['uids']['0'], 'cameras.sfm')
        print(f"'cameras.sfm' found in the StructureFromMotion node.")
    else:
        print("'cameras.sfm' not found. Exiting...")
        sys.exit(1)

    # Get images path
    print("Looking for undistorted images...")
    if 'PrepareDenseScene' in nodes:
        undisto_images_path = os.path.join(cache_folder, graph['PrepareDenseScene_1']['nodeType'], 
                                   graph['PrepareDenseScene_1']['uids']['0'])
        print("Undistorted images found in the PrepareDenseScene node.")
    else:
        undisto_images_path = None
        print("Images will be taken from the 'cameras.sfm' file. They might be distorted.")
    
    # Read 'cameras.sfm' file
    print("Reading 'cameras.sfm'...")
    views_data, intrinsics_data, poses_data = read_cameras_sfm(cameras_sfm_path, undisto_images_path)
    print(f"Found {len(views_data)} views.")
    return views_data, intrinsics_data, poses_data

def write_cameras_sfm(views_data, intrinsics_data, poses_data, cameras_sfm_path):
    with open(cameras_sfm_path, 'w') as f:
        cameras_sfm = {'version': ["1","2","8"], 'views': [], 'intrinsics': [], 'poses': []}
        for view_id, view_data in views_data.items():
            cameras_sfm['views'].append(view_data)
        for intrinsic_id, intrinsic_data in intrinsics_data.items():
            cameras_sfm['intrinsics'].append(intrinsic_data)
        for pose_id, pose_data in poses_data.items():
            cameras_sfm['poses'].append(pose_data)
        json.dump(cameras_sfm, f, indent=4)
    print(f"Cameras written to {cameras_sfm_path}.")