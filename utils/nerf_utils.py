import os
import numpy as np
import json
import shutil
import math

from utils.utils import load_image, save_image, get_view_parameters, _cv_to_gl, _gl_to_cv

def write_nerf_data(views_data, intrinsics_data, poses_data, output_path, bit_depth=16):
    """
    Write NeRF .json file.
    """

    # Create output directory
    images_dir = "images"
    output_images_path = os.path.join(output_path, images_dir)
    os.makedirs(output_images_path, exist_ok=True)

    # Create output object
    output = {
        "w": 0,
        "h": 0,
        "aabb_scale": 1.0,
        # "scale": 0.5,
        # "offset": [ # neus: [-1,1] ngp[0,1]
        #     0.5,
        #     0.5,
        #     0.5
        # ],
        "from_na": True,
        "frames": []
    }

    # Iterate over the views
    for i, (view_id, view_data) in enumerate(views_data.items()):

        # Update width and height
        if i == 0:
            output["w"] = view_data["width"]
            output["h"] = view_data["height"]

        # Load image
        image_path = view_data["path"]
        undisto_image_path = view_data.get("undistortedImagePath", None)
        if undisto_image_path is not None:
            image_path = undisto_image_path    
        image = load_image(image_path)

        # Load mask and save image
        copy_image_bol = False
        mask_path = view_data.get("maskPath", None)
        if mask_path is not None:
            mask = (load_image(mask_path)[:,:,0] > 0.5).astype(np.float32)
        elif image.shape[-1] == 4:
            shutil.copyfile(image_path, os.path.join(output_images_path, f"{i:08d}.png"))
            copy_image_bol = True
        else:
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

        # Save image
        if not copy_image_bol:
            if image.shape[-1] == 3:
                image = np.concatenate([image, mask[...,np.newaxis]], axis=-1)
            elif image.shape[-1] == 4:
                image[:,:,-1] = (image[:,:,-1] > 0.5).astype(np.float32) * mask
            save_image(image, os.path.join(output_images_path, f"{i:08d}.png"), bit_depth=bit_depth)

        # Get view parameters
        K, Rt_c2w, _ = get_view_parameters(view_data, intrinsics_data, poses_data)

        # Add frame to output
        one_frame = {}
        one_frame["file_path"] = os.path.join(images_dir, f"{i:08d}.png")
        one_frame["transform_matrix"] = Rt_c2w.tolist()
        one_frame["intrinsic_matrix"] = K.tolist()
        output["frames"].append(one_frame)

    # Write "transform.json" (handle é, è, à, etc.)
    with open(os.path.join(output_path, "transform.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

def write_neus2_data(views_data, intrinsics_data, poses_data, output_path, bit_depth=16):
    """
    Write NeuS2 .json file.
    Inspired by https://github.com/19reborn/NeuS2/blob/main/tools/data_format_from_neus.py
    """



def write_neuralangelo_data(views_data, intrinsics_data, poses_data, output_path, bit_depth=16, copy_images=True):
    """
    Write NeuralAngelo .json file.
    Inspired by https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/scripts/convert_dtu_to_json.py
    """

    # Create output directory
    images_dir = "image_masked"
    output_images_path = os.path.join(output_path, images_dir)

    # Create output object
    out = {
            "k1": 0.0,  # take undistorted images only
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "is_fisheye": False,
            "frames": []
        }

    # Iterate over the views
    for i, (view_id, view_data) in enumerate(views_data.items()):

        # Load image
        image_path = view_data["path"]
        undisto_image_path = view_data.get("undistortedImagePath", None)
        if undisto_image_path is not None:
            image_path = undisto_image_path    
        image_name = os.path.basename(image_path)

        if copy_images:
            image = load_image(image_path)
            os.makedirs(output_images_path, exist_ok=True)
            mask_path = view_data.get("maskPath", None)
            if mask_path is not None:
                mask = (load_image(mask_path)[:,:,0] > 0.5).astype(np.float32)
            else:
                mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

            if image.shape[-1] == 3:
                image = np.concatenate([image, mask[...,np.newaxis]], axis=-1)
            elif image.shape[-1] == 4:
                image[:,:,-1] = (image[:,:,-1] > 0.5).astype(np.float32) * mask
            save_image(image, os.path.join(output_images_path, image_name), bit_depth=bit_depth)
        else:
            images_dir = os.path.relpath(os.path.dirname(image_path), output_path)
        
        # Get view parameters
        K, c2w_gl, _ = get_view_parameters(view_data, intrinsics_data, poses_data)

        # Add frame to output
        frame = {}
        frame["file_path"] = os.path.join(images_dir, image_name)
        frame["transform_matrix"] = c2w_gl.tolist()
        out["frames"].append(frame)

    # Get intrinsics
    fl_x = float(K[0][0])
    fl_y = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])
    sk_x = float(K[0][1])
    sk_y = float(K[1][0])
    w, h = float(view_data["width"]), float(view_data["height"])

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    pose_data = poses_data[list(poses_data.keys())[0]]
    if "scaleBol" in pose_data["pose"]["scale"]:
        if pose_data["pose"]["scale"]["scaleBol"] == True:
            scale_mat = np.array(pose_data["pose"]["scale"]["scaleMat"], dtype=np.float32).reshape([3,4])
        else:
            scale_mat = np.eye(4)[:3,:4]
    else:
        scale_mat = np.eye(4)[:3,:4]

    out.update({
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "sk_x": sk_x,
        "sk_y": sk_y,
        "w": int(w),
        "h": int(h),
        "aabb_scale": float(np.maximum(np.exp2(np.rint(np.log2(scale_mat[0, 0]))), 1)),  # power of two, for INGP resolution computation
        "sphere_center": [0., 0., 0.],
        "sphere_radius": 1.,
    })

    file_path = os.path.join(output_path, 'transforms.json')
    with open(file_path, "w", encoding="utf-8") as outputfile:
        json.dump(out, outputfile, indent=4)
    print('Writing data to json file: ', file_path)


