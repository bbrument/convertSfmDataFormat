import argparse
import os
import shutil

from utils.metashape_utils import read_metashape_xml
from utils.meshroom_utils import write_cameras_sfm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Metashape project to Meshroom format")
    parser.add_argument("--metashape_project", default="something.xml", help="input path of the .xml file")
    parser.add_argument("--images_folder", default=None, help="images folder")
    parser.add_argument("--sfm_path", default=None, help="Output path of the .sfm file.")
    parser.add_argument("--use-scale-matrix", action="store_true", default=False, help="Use scale matrix.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    METASHAPE_PROJECT_PATH = args.metashape_project
    IMAGES_FOLDER = args.images_folder
    OUTPUT_PATH = args.sfm_path
    if OUTPUT_PATH is None:
        OUTPUT_PATH = os.path.join(os.path.dirname(METASHAPE_PROJECT_PATH), "sfm.json")
    
    # Remove and create output folders
    dirname = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Read IDR data
    views_data, intrinsics_data, poses_data = read_metashape_xml(METASHAPE_PROJECT_PATH, IMAGES_FOLDER)

    # Write Meshroom Sfm file
    write_cameras_sfm(OUTPUT_PATH, views_data, intrinsics_data, poses_data)
