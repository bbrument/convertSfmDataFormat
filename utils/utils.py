import numpy as np
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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

