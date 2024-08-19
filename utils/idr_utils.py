import os
import numpy as np
import cv2
import random

from utils.utils import load_image, save_image

def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def getIntrinsicId(intrinsic_data, intrinsics_data):
    thr = 1e-2
    for intrinsicId, intrinsic in intrinsics_data.items():
        if "pxFocalLength" in intrinsic_data and "pxFocalLength" in intrinsic:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["pxFocalLength"][0]) - float(intrinsic["pxFocalLength"][0])) < thr and \
                abs(float(intrinsic_data["pxFocalLength"][1]) - float(intrinsic["pxFocalLength"][1]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
        elif "focalLength" in intrinsic_data and "focalLength" in intrinsic:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["focalLength"]) - float(intrinsic["focalLength"]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
        else:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
    intrinsicId = intrinsic_data["intrinsicId"]
    return intrinsicId

def getPoseId(pose_data, poses_data):
    thr = 1e-2
    for poseId, pose in poses_data.items():
        if np.allclose(np.array(pose_data["pose"]["transform"]["rotation"], dtype=np.float32), np.array(pose["pose"]["transform"]["rotation"], dtype=np.float32), rtol=thr) and \
            np.allclose(np.array(pose_data["pose"]["transform"]["center"], dtype=np.float32), np.array(pose["pose"]["transform"]["center"], dtype=np.float32), rtol=thr):
            return poseId
    poseId = pose_data["poseId"]
    return poseId

def read_idr_data(idr_folder_path, use_scale_matrix=False):

    # Find folders and cameras.npz
    image_folder = os.path.join(idr_folder_path, "image")
    mask_folder = os.path.join(idr_folder_path, "mask")
    camera_path = os.path.join(idr_folder_path, "cameras.npz")

    if not os.path.exists(image_folder):
        raise ValueError("Image folder not found")
    if not os.path.exists(camera_path):
        raise ValueError("Camera file not found")
    
    # Get image extension and paths
    image_extension = os.path.splitext(os.listdir(image_folder)[0])[1]
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(image_extension)])
    n_images = len(image_paths)

    # Load camera data 
    camera_dict = np.load(camera_path)
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if use_scale_matrix:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    else:
        scale_mats = [np.eye(4, dtype=np.float32) for idx in range(n_images)]

    # Set views, intrinsics and poses
    views_data = {}
    intrinsics_data = {}
    poses_data = {}
    for i, (image_path, world_mat, scale_mat) in enumerate(zip(image_paths, world_mats, scale_mats)):

        # Set view id
        viewId = f"{random.randint(10000000, 1000000000)}"
        while viewId in views_data.keys():
            viewId = f"{random.randint(10000000, 1000000000)}"

        # Load image
        image_path = os.path.abspath(image_path)
        image = load_image(image_path)

        # Get width, height
        width = image.shape[1]
        height = image.shape[0]
        if width/height == 4/3:
            sensor_width = 6.4
            sensor_height = 4.8
        elif width/height == 3/2:
            sensor_width = 36
            sensor_height = 24
        else:
            raise ValueError("Image aspect ratio is not 4:3 or 3:2")

        # Get pose and intrinsics data
        P = world_mat @ scale_mat
        intrinsics, pose = load_K_Rt_from_P(P[:3,:4])

        # Get focal and principal point
        focal_px = float(intrinsics[0, 0])
        focal_py = float(intrinsics[1, 1])
        focal_px_mm = focal_px * sensor_width / width
        focal_py_mm = focal_py * sensor_height / height
        ppx = float(intrinsics[0, 2])
        ppy = float(intrinsics[1, 2])
        cx = ppx - width / 2
        cy = ppy - height / 2

        # Get orientation and center of the camera
        pose[1:3, :] *= -1 # flip y and z axis because Meshroom uses GL coordinates (CV to GL)

        # Set intrinsic and pose data
        intrinsic_data = {
            "intrinsicId": viewId,
            "width": f"{width:0.0f}",
            "height": f"{height:0.0f}",
            "sensorWidth": f"{sensor_width:0.2f}",
            "sensorHeight": f"{sensor_height:0.2f}",
            "serialNumber": "-1",
            "type": "pinhole",
            "initializationMode": "unknown",
            # "pxFocalLength": [f"{focal_px:0.20f}",
            #                 f"{focal_py:0.20f}"
            #                 ],
            # "pxInitialFocalLength": "-1",
            "focalLength": f"{focal_px_mm:0.20f}",
            "initialFocalLength": "-1",
            "pixelRatio": "1",
            "pixelRatioLocked": "1",
            "principalPoint": [f"{cx:0.20f}",
                            f"{cy:0.20f}"
                            ],
            "distortionInitializationMode": "none",
            "distortionParams": ["0", "0", "0"],
            "undistortionOffset": ["0", "0"],
            "undistortionParams": "",
            "distortionType" : "radialk3",
            "undistortionType": "none",
            "locked": "0"
        }
        pose_data = {
            "poseId": viewId,
            "pose": {
                "transform": {
                    "rotation": [f"{x:0.20f}" for x in pose[:3,:3].ravel().tolist()],
                    "center": [f"{x:0.20f}" for x in pose[:3,3].tolist()]
                }
            }
        }

        # Check if intrinsic and pose data already exist
        intrinsicId = getIntrinsicId(intrinsic_data, intrinsics_data)
        poseId = getPoseId(pose_data, poses_data)

        # Update intrinsic and pose ids
        intrinsic_data["intrinsicId"] = intrinsicId
        pose_data["poseId"] = poseId

        # Set view data
        view_data = {
            "viewId": viewId,
            "poseId": poseId,
            "frameId": f"{i+1}",
            "intrinsicId": intrinsicId,
            "resectionId": "",
            "path": image_path,
            "width": f"{width:0.0f}",
            "height": f"{height:0.0f}",
            "metadata" : ""
        }

        # Add data to dictionaries
        intrinsics_data[intrinsicId] = intrinsic_data
        poses_data[poseId] = pose_data
        views_data[viewId] = view_data

    return views_data, intrinsics_data, poses_data

def write_idr_data(views_data, intrinsics_data, poses_data, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    idr_dict = {}
    for i, (view_id, view_data) in enumerate(views_data.items()):
        
        # Get intrinsic and pose
        intrinsic_data = intrinsics_data[view_data["intrinsicId"]]
        pose_data = poses_data[view_data["poseId"]]

        # Get width, height
        width = float(intrinsic_data['width'])
        height = float(intrinsic_data['height'])

        # Get focal length
        if 'pxFocalLength' in intrinsic_data:
            fx = float(intrinsic_data['pxFocalLength'][0])
            fy = float(intrinsic_data['pxFocalLength'][1])
        else:
            sensor_width = float(intrinsic_data['sensorWidth'])
            sensor_height = float(intrinsic_data['sensorHeight'])
            focal_length = float(intrinsic_data['focalLength'])
            fx = focal_length * width / sensor_width
            fy = focal_length * height / sensor_height

        # Get principal point
        cx = width / 2 + float(intrinsic_data['principalPoint'][0])
        cy = height / 2 + float(intrinsic_data['principalPoint'][1])

        # Get intrinsics matrix
        K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        # Get rotation matrix and center
        R_c2w = np.array(pose_data['pose']['transform']['rotation'], dtype=np.float32).reshape([3,3]) # orientation
        center = np.expand_dims(np.array(pose_data['pose']['transform']['center'], dtype=np.float32), axis=1) # center
        Rt_c2w = np.eye(4)
        Rt_c2w[:3,:3] = R_c2w
        Rt_c2w[:3,3] = center[:,0]
        Rt_c2w[1:3, :] *= -1 # flip y and z axis because Meshroom uses GL coordinates (GL to CV)
        idr_dict['world_mat_%d'%i] = K @ np.linalg.inv(Rt_c2w)

        # Load image
        image_path = view_data["path"]
        undisto_image_path = view_data.get("undistortedImagePath", None)
        if undisto_image_path is not None:
            image = load_image(undisto_image_path)
        else:
            image = load_image(image_path)

        # Save image and mask
        output_image_path = os.path.join(save_path, "image")
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)
        save_image(image[:, :, :3], os.path.join(output_image_path, f"{i:08d}.png"), bit_depth=16)
        if image.shape[-1] == 4:
            output_mask_path = os.path.join(save_path, "mask")
            if not os.path.exists(output_mask_path):
                os.makedirs(output_mask_path)
            cv2.imwrite(os.path.join(output_mask_path, f"{i:08d}.png"), (image[:, :, 3]*255).astype(np.uint8))

    # Save npz file
    output_npz_path = os.path.join(save_path, "cameras.npz")
    np.savez(output_npz_path,**idr_dict)