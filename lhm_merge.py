import argparse
from pathlib import Path
import torch
import numpy as np

def merge_smpl_parms(root_dir, split):
    root = Path(root_dir)
    old_path = root / split / "smpl_parms_orig.pth"
    new_path = root / "poses_optimized_lhm.npz"
    save_path = root / split / "smpl_parms.pth"

    # 讀取舊資料
    print(f"Loading old pth from: {old_path}")
    old_data = torch.load(old_path, map_location="cpu")

    # 讀取新資料
    print(f"Loading new npz from: {new_path}")
    new_npz = np.load(new_path)
    if "thetas" in new_npz:
        print("Detected format: [thetas]")
        new_theta = new_npz["thetas"].astype(np.float32)  # shape (N, 72)
        new_trans = new_npz["transl"].astype(np.float32)  # shape (N, 3)
    elif "global_orient" in new_npz and "body_pose" in new_npz:
        print("Detected format: [global_orient + body_pose]")
        new_theta = np.concatenate([
            new_npz["global_orient"],       # shape (N, 3)
            new_npz["body_pose"]            # shape (N, 69)
        ], axis=1).astype(np.float32)       # → shape (N, 72)
        new_trans = new_npz["transl"].astype(np.float32)  # shape (N, 3)
    else:
        raise ValueError("Unsupported pose format in npz.")

    # 合併
    print("Merging tensors...")
    merged = {
        "beta": old_data["beta"],  # 保留原始 beta，如需替換可改成 torch.from_numpy(new_beta)
        "body_pose": torch.cat([old_data["body_pose"], torch.from_numpy(new_theta)], dim=0),
        "trans": torch.cat([old_data["trans"], torch.from_numpy(new_trans)], dim=0),
        "lhm_start": old_data["body_pose"].shape[0],  # 新增 lhm_start
    }
    print("lhm_start:" , old_data["body_pose"].shape[0])
    # 儲存
    print(f"Saving merged pth to: {save_path}")
    torch.save(merged, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing train/test and lhm_pose folders")
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True, help="Choose split: train or test")
    args = parser.parse_args()

    merge_smpl_parms(args.root_dir, args.split)
