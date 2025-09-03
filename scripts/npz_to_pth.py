import os
import torch
import argparse
import numpy as np
from os.path import join


def load_smpl_param(path, data_list, return_theta=True):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

    if not return_theta:
        return {
            "betas": smpl_params["betas"].astype(np.float32),
            "body_pose": smpl_params["body_pose"].astype(np.float32),
            "global_orient": smpl_params["global_orient"].astype(np.float32),
            "transl": smpl_params["transl"].astype(np.float32),
        }

    theta = np.zeros((len(data_list), 72), dtype=np.float32)
    trans = np.zeros((len(data_list), 3), dtype=np.float32)

    for idx, _ in enumerate(data_list):
        theta[idx, :3] = smpl_params["global_orient"][idx].astype(np.float32)
        theta[idx, 3:] = smpl_params["body_pose"][idx].astype(np.float32)
        trans[idx, :] = smpl_params["transl"][idx].astype(np.float32)

    return {
        "beta": torch.from_numpy(smpl_params["betas"].reshape(1, 10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }


def main():
    parser = argparse.ArgumentParser(description="Process SMPL and camera parameters.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset folder containing images and SMPL/camera params")
    parser.add_argument("--cam_npz", type=str, default="cam_parms_lhm.npz",
                        help="Camera parameter .npz filename inside data_path")
    parser.add_argument("--pose_npz", type=str, default="poses_optimized_lhm.npz",
                        help="SMPL pose parameter .npz filename inside data_path")
    args = parser.parse_args()

    data_path = args.data_path

    # Images
    all_image_path = join(data_path, "images")
    data_list = os.listdir(all_image_path)
    scene_length = len(data_list)
    print(f"Number of images: {scene_length}")

    # Camera params
    cam_file = join(data_path, args.cam_npz)
    camera = np.load(cam_file)
    intrinsic = np.array(camera["intrinsic"])
    extrinsic = np.array(camera["extrinsic"])
    cam_all = {"intrinsic": intrinsic, "extrinsic": extrinsic}
    np.savez(join(data_path, "cam_parms.npz"), **cam_all)

    # SMPL params
    pose_file = join(data_path, args.pose_npz)
    smpl_params = load_smpl_param(pose_file, data_list)
    torch.save(smpl_params, join(data_path, "smpl_parms.pth"))

    print(f"Processed SMPL ({args.pose_npz}) and camera ({args.cam_npz}) params saved to {data_path}")


if __name__ == "__main__":
    main()

