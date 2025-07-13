import torch
import os
from os.path import join
import argparse


def export_pose_and_transl(net_path, smpl_parms_path, output_path):
    net_name = 'net.pth'

    # Load saved model
    saved_model_state = torch.load(join(net_path, net_name))
    print('✅ Loaded model from:', join(net_path, net_name))

    # Load smpl_parms.pth just for reference (not directly used)
    _ = torch.load(join(smpl_parms_path, 'smpl_parms.pth'))

    # Setup Embedding model
    num_training_frames = len(os.listdir(join(smpl_parms_path, 'raw_images')))
    pose = torch.nn.Embedding(num_training_frames, 72, sparse=True).cuda()
    transl = torch.nn.Embedding(num_training_frames, 3, sparse=True).cuda()

    # Load weights
    pose.load_state_dict(saved_model_state["pose"], strict=False)
    transl.load_state_dict(saved_model_state["transl"], strict=False)

    # Define test frame indices (same as training split logic)
    num_val = num_training_frames // 5
    length = int(1 / num_val * num_training_frames)
    offset = length // 2
    val_list = list(range(num_training_frames))[offset::length]
    test_list = val_list[:len(val_list) // 2]

    # Extract test pose & trans
    body_pose = []
    trans = []
    for idx in test_list:
        idx_tensor = torch.tensor([idx], device='cuda')
        body_pose.append(pose(idx_tensor).squeeze(0).cpu().detach())  # (72,)
        trans.append(transl(idx_tensor).squeeze(0).cpu().detach())    # (3,)

    # Stack and save
    test_data = {
        'body_pose': torch.stack(body_pose),  # (N, 72)
        'trans': torch.stack(trans)           # (N, 3)
    }
    torch.save(test_data, output_path)
    print(f"✅ Saved test poses to: {output_path}")
    print(f"  body_pose shape: {test_data['body_pose'].shape}")
    print(f"  trans shape: {test_data['trans'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_path', required=True, help='Path to saved net folder (should contain net.pth)')
    parser.add_argument('--smpl_parms_path', required=True, help='Path to folder containing images and smpl_parms.pth')
    parser.add_argument('--output_path', required=True, help='Output file path to save smpl_parms_pred.pth')

    args = parser.parse_args()
    export_pose_and_transl(args.net_path, args.smpl_parms_path, args.output_path)
