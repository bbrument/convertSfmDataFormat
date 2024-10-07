import argparse
import os
import shutil

from utils.metashape_utils import read_metashape_project
from utils.idr_utils import write_idr_data

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Metashape project to IDR format")
    parser.add_argument("--metashape_project", default="something.xml", help="input path of the .xml file")
    parser.add_argument("--mask_folder", default=None, help="mask folder")
    parser.add_argument("--output_path", default=None, help="output path")
    parser.add_argument("--bit_depth", default=16, type=int, help="bit depth of the images")
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
        OUTPUT_PATH = os.path.join(os.path.dirname(MESHROOM_PROJECT_PATH), "idr")
    BIT_DEPTH = args.bit_depth
        
    # Remove and create output folders
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # Read meshroom project
    views_data, intrinsics_data, poses_data, all_rest_data = read_meshroom_project(MESHROOM_PROJECT_PATH, cameras_sfm_path=CAMERAS_SFM_PATH, masks_folder=MASK_FOLDER)

    # Output images and camera parameters in the IDR format
    write_idr_data(views_data, intrinsics_data, poses_data, OUTPUT_PATH, bit_depth=BIT_DEPTH)