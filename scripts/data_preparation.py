
import os
import sys
import shutil
import torch
import argparse
import numpy as np
import trimesh
from os.path import join

BASE_DIR = os.getcwd()

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
    for iter, idx in enumerate(data_list):
        theta[iter, :3] = smpl_params["global_orient"][idx].astype(np.float32)
        theta[iter, 3:] = smpl_params["body_pose"][idx].astype(np.float32)
        trans[iter, :] = smpl_params["transl"][idx].astype(np.float32)

    return {
        "beta": torch.from_numpy(smpl_params["betas"].reshape(1,10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }

def save_obj(data_path):
    sys.path.append(BASE_DIR)
    from submodules import smplx
    from arguments import smpl_cpose_param
    smpl_data = torch.load( data_path + '/smpl_parms.pth')
    smpl_model = smplx.SMPL(model_path ='assets/smpl_files/smpl',batch_size = 1)
    cano_dir = os.path.join(data_path,)


    cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                            global_orient=smpl_cpose_param[:, :3],
                            transl = torch.tensor([[0, 0.30, 0]]),
                            # global_orient=cpose_param[:, :3],
                            body_pose=smpl_cpose_param[:, 3:],
                            )

    ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
    joint_mat = cano_smpl.A
    print(joint_mat.shape)
    torch.save(joint_mat ,join(cano_dir, 'smpl_cano_joint_mat.pth'))


    mesh = trimesh.Trimesh(ori_vertices, smpl_model.faces, process=False)
    mesh.export('%s/%s.obj' % (cano_dir, 'cano_smpl'))

def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id

def save_npz(data_path, res=512, uv_template_fn=None):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    result = {}
    body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smpl'), process=False)

    if res==128:
        posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
        result['posmap128'] = posmap128   
    elif res == 256:
    
        posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
        result['posmap256'] = posmap256

    elif res == 384:
    
        posmap384, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=384)
        result['posmap384'] = posmap384

    else:
        posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
        result['posmap512'] = posmap512

    save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smpl'))
    np.savez(save_fn, **result)
    print("Pose map saved to", data_path)

def main(args):
    data_path = os.path.join(BASE_DIR, args.data_path.lstrip("/"))
    resolution = args.resolution

    image_dir = join(data_path, 'images')
    mask_dir = join(data_path, 'masks')
    scene_length = len(os.listdir(image_dir))

    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    train_list = sorted(train_list+test_list+val_list)

    train_split_name, test_split_name = [], []
    for idx, fname in enumerate(sorted(os.listdir(image_dir))):
        if idx in train_list: train_split_name.append(fname)
        if idx in test_list: test_split_name.append(fname)

    out_path, test_path = join(data_path, 'train'), join(data_path, 'test')
    for p in [join(out_path, 'images'), join(out_path, 'masks'),
              join(test_path, 'images'), join(test_path, 'masks')]:
        os.makedirs(p, exist_ok=True)

    camera = np.load(join(data_path, "cameras.npz"))
    np.savez(join(out_path, 'cam_parms.npz'), **camera)
    np.savez(join(test_path, 'cam_parms.npz'), **camera)

    torch.save(load_smpl_param(join(data_path, "poses_optimized.npz"), train_list), join(out_path, 'smpl_parms.pth'))
    torch.save(load_smpl_param(join(data_path, "poses_optimized.npz"), test_list), join(test_path, 'smpl_parms.pth'))

    for name in train_split_name:
        shutil.copy(join(image_dir, name), join(out_path, 'images', name))
        shutil.copy(join(mask_dir, name), join(out_path, 'masks', name))
    for name in test_split_name:
        shutil.copy(join(image_dir, name), join(test_path, 'images', name))
        shutil.copy(join(mask_dir, name), join(test_path, 'masks', name))

    print("Split done. Now saving SMPL obj and posemap...")

    uv_template_fn = join(BASE_DIR, 'assets/template_mesh_smpl_uv.obj')
    for split in [out_path, test_path]:
        save_obj(split)
        save_npz(split, res=resolution, uv_template_fn=uv_template_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path (contains images/, masks/, cameras.npz, poses_optimized.npz)")
    parser.add_argument("--resolution", type=int, default=512, help="UV posemap resolution (default=512)")
    args = parser.parse_args()
    main(args)
