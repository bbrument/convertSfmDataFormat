import numpy as np
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def _cv_to_gl(cv_matrix):
    """
    Convert a camera matrix from OpenCV convention to OpenGL convention.
    """
    cv_to_gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    gl_matrix = cv_to_gl @ cv_matrix
    return gl_matrix

def _gl_to_cv(gl_matrix):
    """
    Convert a camera matrix from OpenGL convention to OpenCV convention.
    """
    return _cv_to_gl(gl_matrix)

def load_image_opencv(path):
    
    # Load image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Convert to float32
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    image = np.float32(image) / (2**bit_depth - 1)

    # Handle alpha channel if present
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_image_exr(path):

    # Load image
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image = np.float32(image)

    # Convert from linear to sRGB
    def linear_to_srgb(linear):
        a = 0.055
        threshold = 0.0031308
        srgb = np.where(linear <= threshold,
                        linear * 12.92,
                        (1.0 + a) * np.power(linear, 1.0 / 2.4) - a)
        return np.clip(srgb, 0.0, 1.0)

    image = linear_to_srgb(image)

    # Handle alpha channel if present
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_image(path):

    # If .png, .jpg, or .jpeg, load with OpenCV
    if path.endswith((".png", ".jpg", ".jpeg")):
        return load_image_opencv(path)
    elif path.endswith(".exr"):
        return load_image_exr(path)
    

def save_image(image, path, bit_depth=8):

    # Convert to uint8 or uint16
    image = (image * np.float32(2**bit_depth - 1))
    if bit_depth == 8:
        image = image.astype(np.uint8)
    elif bit_depth == 16:
        image = image.astype(np.uint16)

    # Handle alpha channel if present
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def save_mask(mask, path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask)


def get_view_parameters(view, intrinsics, poses):

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
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    # Get rotation matrix and center in OpenGL convention
    R_c2w = np.array(pose['pose']['transform']['rotation'], dtype=np.float32).reshape([3,3]) # orientation
    center = np.expand_dims(np.array(pose['pose']['transform']['center'], dtype=np.float32), axis=1) # center
    Rt_c2w_gl = np.eye(4)
    Rt_c2w_gl[:3,:3] = R_c2w
    Rt_c2w_gl[:3,3] = center[:,0]

    return K, Rt_c2w_gl, path