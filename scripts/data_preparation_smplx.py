import os
import sys
import shutil
import argparse
import numpy as np
import torch
from os.path import join
import trimesh

sys.path.append('../')
from submodules import smplx
from arguments import smplx_cpose_param, smpl_cpose_param
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori


def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    from posmap_generator.lib.renderer.gl.pos_render import PosRender
    rndr = PosRender(width=img_size, height=img_size)
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)
    rndr.display()
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3].reshape(-1)
    uv_pos = uv_pos[:, :, :3].reshape(-1, 3)
    rendered_pos = uv_pos[uv_mask != 0.0]
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1
    assert len(face_id) == len(rendered_pos)
    return uv_pos.reshape(img_size, img_size, 3), uv_mask, face_id


def save_obj_smplx(data_path, name='smpl_parms.pth'):
    smplx_data = torch.load(join(data_path, name))
    smplx_model = smplx.SMPLX(model_path='./assets/smpl_files/smplx', gender='neutral', use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1)
    cano_dir = data_path
    live_smplx = smplx_model.forward(
        betas=smplx_data['beta'][0][None],
        transl=torch.tensor([[0, 0.35, 0]]),
        global_orient=smplx_cpose_param[:, :3],
        body_pose=smplx_cpose_param[:, 3:66],
        jaw_pose=smplx_cpose_param[:, 66:69],
        leye_pose=smplx_cpose_param[:, 69:72],
        reye_pose=smplx_cpose_param[:, 72:75],
        left_hand_pose=smplx_cpose_param[:, 75:120],
        right_hand_pose=smplx_cpose_param[:, 120:]
    )
    vertices = live_smplx.vertices.detach().cpu().numpy().squeeze()
    joint_mat = live_smplx.A
    torch.save(joint_mat, join(cano_dir, 'smplx_cano_joint_mat.pth'))
    mesh = trimesh.Trimesh(vertices, smplx_model.faces, process=False)
    mesh.export(f'{cano_dir}/cano_smplx.obj')


def save_npz(data_path, res=128, uv_template_fn=None):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    body_mesh = trimesh.load(join(data_path, 'cano_smplx.obj'), process=False)
    result = {}
    if res == 128:
        result['posmap128'], _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
    elif res == 256:
        result['posmap256'], _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
    else:
        result['posmap512'], _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
    save_fn = join(data_path, f'query_posemap_{res}_cano_smplx.npz')
    np.savez(save_fn, **result)


def load_smplx_param(path, data_list):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    theta = np.zeros((len(data_list), 165), dtype=np.float32)
    trans = np.zeros((len(data_list), 3), dtype=np.float32)
    for i, idx in enumerate(data_list):
        theta[i, :3] = smpl_params["global_orient"][idx]
        theta[i, 3:] = smpl_params["body_pose"][idx]
        trans[i, :] = smpl_params["transl"][idx]
    return {
        "beta": torch.from_numpy(smpl_params["betas"].reshape(1, 10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }


def main(args):
    data_path = args.data_path
    resolution = args.resolution
    uv_template_fn =  './assets/template_mesh_smplx_uv.obj'

    all_image_path = join(data_path, 'images')
    all_mask_path = join(data_path, 'masks')
    scene_length = len(os.listdir(all_image_path))

    # InstantAvatar-style split
    num_val = scene_length // 5
    offset = (scene_length // num_val) // 2
    val_list = list(range(scene_length))[offset::(scene_length // num_val)]
    train_list = sorted(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    train_list = sorted(train_list + val_list + test_list)

    image_names = sorted(os.listdir(all_image_path))
    train_names = [image_names[i] for i in train_list]
    test_names = [image_names[i] for i in test_list]

    for split, names, ids in zip(['train', 'test'], [train_names, test_names], [train_list, test_list]):
        split_path = join(data_path, split)
        os.makedirs(join(split_path, 'images'), exist_ok=True)
        os.makedirs(join(split_path, 'masks'), exist_ok=True)
        for name in names:
            shutil.copy(join(all_image_path, name), join(split_path, 'images', name))
            shutil.copy(join(all_mask_path, name), join(split_path, 'masks', name))
        smpl_params = load_smplx_param(join(data_path, "poses_optimized.npz"), ids)
        torch.save(smpl_params, join(split_path, 'smpl_parms.pth'))

    print('Saving SMPL-X canonical mesh and joint matrix...')
    save_obj_smplx(join(data_path, 'train'))

    print(f'Saving posemap at resolution {resolution}...')
    save_npz(join(data_path, 'train'), resolution, uv_template_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset (should contain images/, masks/, poses_optimized.npz)')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of pose map (128, 256, 512)')
    args = parser.parse_args()
    main(args)

