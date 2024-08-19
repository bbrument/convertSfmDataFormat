import argparse
import os
import shutil

from utils.meshroom_utils import read_meshroom_project
from utils.idr_utils import write_idr_data

def parse_args():
    parser = argparse.ArgumentParser(description="convert a meshroom sfm data file to idr format")
    parser.add_argument("--meshroom_project", default="something.mg", help="input path of the .mg file")
    parser.add_argument("--output_path", default=None, help="output path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    MESHROOM_PROJECT_PATH = args.meshroom_project
    if args.output_path is None:
        OUTPUT_PATH = os.path.join(os.path.dirname(MESHROOM_PROJECT_PATH), "idr")
    else:
        OUTPUT_PATH = args.output_path
    
    # Remove and create output folders
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # Read meshroom project
    views_data, intrinsics_data, poses_data = read_meshroom_project(MESHROOM_PROJECT_PATH)

    # Filter some ids
    list_ids_to_keep = [1766124410, 1016283824, 1037767162, 927099987, 1828908878]
    new_views_data = {}
    for id in list_ids_to_keep:
        new_views_data[str(id)] = views_data[str(id)]
    views_data = new_views_data

    # Output images and camera parameters in the IDR format
    print("Output images and camera parameters...")
    write_idr_data(views_data, intrinsics_data, poses_data, OUTPUT_PATH)

    # Save .mat file with K and poses
    # import scipy.io
    # mat_dict = {'K':[], 'poses':[]}
    # for i, (view_id, view_data) in enumerate(data.items()):
    #     K = view_data["intrinsics"]
    #     c2w = view_data["c2w"]
    #     if i == 0:
    #         mat_dict['K'] = np.array(K, dtype=np.float64)
    #     mat_dict['poses'].append(np.linalg.inv(c2w))
    # mat_dict['poses'] = np.array(mat_dict['poses'], dtype=np.float64)
    # scipy.io.savemat(os.path.join(OUTPUT_PATH, 'cameras.mat'), mat_dict)
    
    