
import os
import torch
import shutil
import numpy as np
import numpy as np
from os.path import join


def load_smpl_param(path, data_list, return_thata=True):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

    if not return_thata:
        return {
        "betas": smpl_params["betas"].astype(np.float32),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

    theta = np.zeros((len(data_list), 72)).astype(np.float32)
    trans  = np.zeros((len(data_list), 3)).astype(np.float32)
    iter = 0
    for idx, _ in enumerate(data_list):
        theta[iter, :3] = smpl_params["global_orient"][idx].astype(np.float32)
        theta[iter, 3:] = smpl_params["body_pose"][idx].astype(np.float32)
        trans[iter, :] = smpl_params["transl"][idx].astype(np.float32)

        iter +=1

    return {
        "beta": torch.from_numpy(smpl_params["betas"].mean(axis=0).reshape(1,10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }




data_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/demo/live'




all_image_path = join(data_path, 'images')

data_list = os.listdir(all_image_path)
scene_length = len(data_list)
print(scene_length)

camera = np.load(join(data_path, "cam_parms_lhm.npz"))
intrinsic = np.array(camera["intrinsic"])
extrinsic = np.array(camera["extrinsic"])
cam_all = {} 

cam_all['intrinsic'] = intrinsic
cam_all['extrinsic'] = extrinsic
np.savez(join(data_path, 'cam_parms.npz'), **cam_all)


smpl_params = load_smpl_param(join(data_path, "poses_optimized_lhm.npz"), data_list)

torch.save(smpl_params ,join(data_path, 'smpl_parms.pth'))


