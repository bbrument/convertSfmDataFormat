import argparse
import os
import shutil

from utils.meshroom_utils import read_meshroom_project, write_cameras_sfm
from utils.nerf_utils import write_nerf_data

def parse_args():
    parser = argparse.ArgumentParser(description="Converts a Meshroom project or .sfm file to a Instant-ngp or NeuS2 .json file.")
    parser.add_argument("--meshroom_project", default="something.mg", help="input path of the .mg file")
    parser.add_argument("--cameras_sfm", default=None, help="input path of the cameras.sfm file")
    parser.add_argument("--mask_folder", default=None, help="mask folder")
    parser.add_argument("--output_format", default="neuralangelo", help="Output format of the .json file.")
    parser.add_argument("--output_path", default=None, help="output path")
    parser.add_argument("--bit_depth", default=16, type=int, help="Bit depth of the images.")
    parser.add_argument("--copy_images", action="store_true", help="Copy images to the output folder.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    MESHROOM_PROJECT_PATH = args.meshroom_project
    CAMERAS_SFM_PATH = args.cameras_sfm
    MASK_FOLDER = args.mask_folder
    OUTPUT_PATH = args.output_path
    if args.output_path is None:
        OUTPUT_PATH = os.path.join(os.path.dirname(MESHROOM_PROJECT_PATH), "nerf")
    BIT_DEPTH = args.bit_depth
    COPY_IMAGES = args.copy_images

    # Read meshroom project
    views_data, intrinsics_data, poses_data, all_rest_data = read_meshroom_project(MESHROOM_PROJECT_PATH, cameras_sfm_path=CAMERAS_SFM_PATH, masks_folder=MASK_FOLDER)

    # Write Instant-ngp or NeuS2 .json file
    if args.output_format == "neuralangelo":
        write_neuralangelo_data(views_data, intrinsics_data, poses_data, OUTPUT_PATH, bit_depth=BIT_DEPTH, copy_images=COPY_IMAGES)
    elif args.output_format == "neus2":
        write_neus2_data(views_data, intrinsics_data, poses_data, OUTPUT_PATH, bit_depth=BIT_DEPTH, copy_images=COPY_IMAGES)
    else:
        write_nerf_data(views_data, intrinsics_data, poses_data, OUTPUT_PATH, bit_depth=BIT_DEPTH, copy_images=COPY_IMAGES)
    