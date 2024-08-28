import os
import numpy as np
import json
import sys


def read_cameras_sfm(cameras_sfm_path, undisto_images_folder=None, masks_folder=None):
    
    # Read cameras.sfm
    with open(cameras_sfm_path, 'r') as f:
        cameras_sfm = json.load(f)

    # Parse cameras.sfm
    views = cameras_sfm['views']
    intrinsics = cameras_sfm['intrinsics']
    poses = cameras_sfm['poses']

    # Get version, featuresFolders, matchesFolders, structure
    version = None
    featuresFolders = None
    matchesFolders = None
    structure = None
    if 'version' in cameras_sfm:
        version = cameras_sfm['version']
    if 'featuresFolders' in cameras_sfm:
        featuresFolders = cameras_sfm['featuresFolders']
    if 'matchesFolders' in cameras_sfm:
        matchesFolders = cameras_sfm['matchesFolders']
    if 'structure' in cameras_sfm:
        structure = cameras_sfm['structure']

    # List all undistorted images and masks
    if undisto_images_folder is not None:
        all_undisto_paths = [f for f in os.listdir(undisto_images_folder) if f.endswith('.png') or f.endswith('.exr')]
    if masks_folder is not None:
        all_mask_paths = [f for f in os.listdir(masks_folder) if f.endswith('.png')]

    # Get views, intrinsics and poses data
    views_data = {}
    intrinsics_data = {}
    poses_data = {}
    for view in views:

        # Get id and image name
        view_id = view['viewId']
        image_name = os.path.basename(view['path']).split('.')[0]

        # Add view data
        views_data[view['viewId']] = view

        # Check if undistorted image exists
        if all_undisto_paths is not None:
            for undisto_path in all_undisto_paths:
                if image_name.lower() in undisto_path.lower() or view_id in undisto_path.lower():
                    view['undistortedImagePath'] = os.path.join(undisto_images_folder, undisto_path)
                    break
        else:
            view['undistortedImagePath'] = None

        # Check if mask exists
        if masks_folder is not None:
            for mask_path in all_mask_paths:
                if image_name.lower() in mask_path.lower() or view_id in mask_path.lower():
                    view['maskPath'] = os.path.join(masks_folder, mask_path)
                    break
        else:
            view['maskPath'] = None
    
    for intrinsic in intrinsics:
        intrinsics_data[intrinsic['intrinsicId']] = intrinsic
    
    for pose in poses:
        poses_data[pose['poseId']] = pose

    return views_data, intrinsics_data, poses_data, [version, featuresFolders, matchesFolders, structure]

def read_meshroom_project(meshroom_project_path, cameras_sfm_path=None, masks_folder=None):
    with open(meshroom_project_path, 'r') as f:
        data = json.load(f)

    # Get nodes and graph
    cache_folder = os.path.join(os.path.dirname(meshroom_project_path), 'MeshroomCache')
    nodeNames = data['header']['nodesVersions']
    graph = data['graph']
    nodes = list(graph.keys())

    # Get 'cameras.sfm' path
    print("Looking for 'cameras.sfm'...")
    if cameras_sfm_path is not None:
        print(f"'cameras.sfm' found in the input path.")
    elif 'ConvertSfMFormat' in nodeNames:
        sorted_nodes = [node for node in nodes if 'ConvertSfMFormat' in node]
        sorted_nodes.sort(key=lambda x: int(x.split('_')[-1]))
        last_node = sorted_nodes[-1]
        cameras_sfm_path = os.path.join(cache_folder, graph[last_node]['nodeType'], 
                                        graph[last_node]['uids']['0'], 'sfm.sfm')
        if not os.path.exists(cameras_sfm_path):
            cameras_sfm_path = os.path.join(cache_folder, graph[last_node]['nodeType'], 
                                            graph[last_node]['uids']['0'], 'sfm.json')
            print(f"'sfm.json' found in the last ConvertSfMFormat node.")
        else:
            print(f"'sfm.sfm' found in the last ConvertSfMFormat node.")
        
    elif 'StructureFromMotion' in nodeNames:
        sorted_nodes = [node for node in nodes if 'StructureFromMotion' in node]
        sorted_nodes.sort(key=lambda x: int(x.split('_')[-1]))
        last_node = sorted_nodes[-1]
        cameras_sfm_path = os.path.join(cache_folder, graph[last_node]['nodeType'], 
                                        graph[last_node]['uids']['0'], 'cameras.sfm')
        print(f"'cameras.sfm' found in the last StructureFromMotion node.")

    else:
        print("'cameras.sfm' not found. Exiting...")
        sys.exit(1)

    # Get images path
    print("Looking for undistorted images...")
    if 'PrepareDenseScene' in nodeNames:
        sorted_nodes = [node for node in nodes if 'PrepareDenseScene' in node]
        sorted_nodes.sort(key=lambda x: int(x.split('_')[-1]))
        last_node = sorted_nodes[-1]
        undisto_images_path = os.path.join(cache_folder, graph[last_node]['nodeType'], 
                                   graph[last_node]['uids']['0'])
        print("Undistorted images found in the PrepareDenseScene node.")
    else:
        undisto_images_path = None
        print("Images will be taken from the 'cameras.sfm' file. They might be distorted.")
    
    # Read 'cameras.sfm' file
    print("Reading 'cameras.sfm'...")
    views_data, intrinsics_data, poses_data, all_rest_data = read_cameras_sfm(cameras_sfm_path, undisto_images_path, masks_folder)
    print(f"Found {len(views_data)} views.")
    return views_data, intrinsics_data, poses_data, all_rest_data

def write_cameras_sfm(views_data, intrinsics_data, poses_data, all_rest_data, cameras_sfm_path):
    
    # Initialize cameras_sfm
    cameras_sfm = {}
    for i, data in enumerate(all_rest_data[:3]):
        if data is not None:
            if i == 0:
                cameras_sfm['version'] = data
            elif i == 1:
                cameras_sfm['featuresFolders'] = data
            elif i == 2:
                cameras_sfm['matchesFolders'] = data
    cameras_sfm['views'] = []
    cameras_sfm['intrinsics'] = []
    cameras_sfm['poses'] = []
    if all_rest_data[3] is not None:
        cameras_sfm['structure'] = all_rest_data[3]

    # Write views, intrinsics and poses data
    for view_id, view_data in views_data.items():
        cameras_sfm['views'].append(view_data)
    for intrinsic_id, intrinsic_data in intrinsics_data.items():
        cameras_sfm['intrinsics'].append(intrinsic_data)
    for pose_id, pose_data in poses_data.items():
        cameras_sfm['poses'].append(pose_data)
    
    # Write cameras.sfm (handle é, è, à, etc.)
    with open(cameras_sfm_path, 'w', encoding='utf-8') as f:
        json.dump(cameras_sfm, f, indent=4, ensure_ascii=False)
    print(f"Cameras written to {cameras_sfm_path}.")