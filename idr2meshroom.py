import argparse
import os

from utils.idr_utils import read_idr_data
from utils.meshroom_utils import write_cameras_sfm

def parse_args():
    parser = argparse.ArgumentParser(description="Converts an IDR format folder to a Meshroom Sfm data file.")
    parser.add_argument("--idr_folder_path", default="data/", help="Input path of the IDR folder.")
    parser.add_argument("--sfm_path", default=None, help="Output path of the .sfm file.")
    parser.add_argument("--use-scale-matrix", action="store_true", default=False, help="Use scale matrix.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    IDR_FOLDER_PATH = args.idr_folder_path
    OUTPUT_PATH = args.sfm_path
    if OUTPUT_PATH is None:
        OUTPUT_PATH = os.path.join(IDR_FOLDER_PATH, "sfm.json")
    USE_SCALE_MATRIX = args.use_scale_matrix
    
    # Remove and create output folders
    dirname = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Read IDR data
    views_data, intrinsics_data, poses_data = read_idr_data(IDR_FOLDER_PATH, USE_SCALE_MATRIX)

    # Write Meshroom Sfm file
    write_cameras_sfm(views_data, intrinsics_data, poses_data, OUTPUT_PATH)


