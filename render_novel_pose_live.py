import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import  AvatarModel
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.io import write_video
import romp
from pathlib import Path
import imageio.v2 as imageio

def load_smpl_param(smpl_params, return_thata=True):
    #smpl_params = dict(np.load(str(path)))
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

    theta = np.zeros((1, 72)).astype(np.float32)
    trans  = np.zeros((1, 3)).astype(np.float32)
    theta[0, :3] = smpl_params["global_orient"][0].astype(np.float32)
    theta[0, 3:] = smpl_params["body_pose"][0].astype(np.float32)
    trans[0, :] = smpl_params["trans"][0].astype(np.float32)


    return {
        "beta": torch.from_numpy(smpl_params["betas"].reshape(1,10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }

def render_sets(model, net, opt, epoch:int, args):
    settings = romp.main.default_settings 
    settings.mode = 'video'
    #settings.show = True
    settings.t = True
    settings.sc = 1
    #settings.onnx = True
    settings.show_largest = True
    romp_model = romp.ROMP(settings)
    
    frame_paths= sorted(os.listdir(args.input))
    #print(frame_paths)
    #background = cap.read()
    iteration = 0
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()
        avatarmodel.load(epoch)
        gif_list=[]
        video_list=[]
        for frame_path in tqdm(frame_paths):
            frame_path=args.input+'/'+frame_path
            frame = cv2.imread(frame_path)
            result = romp_model(frame)
            #if not ret:
             #   print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
              #  break
            ##################如果人的input size太小render出來會腫腫的 2025.05.03#################################
            ####################################################################################################
            #frame = cv2.resize(frame, (1080, 1080), interpolation=cv2.INTER_AREA)
            #frame = cv2.imread("./test.jpg")
            #print("cv2 show")
            #cv2.startWindowThread()
            
            #cv2.imshow("Gaussian Avatar", frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            

            background = cv2.imread('./demo/background/images/00000000.png')
            #cv2.imshow("background", background)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
            
            #background = cv2.resize(background, (1920, 1440), interpolation=cv2.INTER_AREA)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                #background = cv2.resize(background, (1080, 1080), interpolation=cv2.INTER_AREA)
            background = background.transpose(2,0,1)
            background = background / 255.
            background = torch.tensor(background, dtype=torch.float32, device="cuda")
            background=None

                
            if result is None:
                continue
            if result["body_pose"].shape[0] > 1:
                result = {k: v[0:1] for k, v in result.items()}
            result = {
                "betas": result["smpl_betas"].mean(axis=0),
                "global_orient": result["smpl_thetas"][:, :3],
                "body_pose": result["smpl_thetas"][:, 3:],
                "trans": result["cam_trans"],
            }
            novel_pose = load_smpl_param(result)
            #print(result["trans"])

            fov = 60
            f = max(frame.shape[:2]) / 2 * 1 / np.tan(np.radians(fov/2))
            K = np.eye(3)
            K[0, 0] = K[1, 1] = f
            K[0, 2] = frame.shape[1] / 2
            K[1, 2] = frame.shape[0] / 2
            #K[0, 2] = 1080.0 / 2
            #K[1, 2] = 1080.0 / 2
            T = np.eye(4)
            #T[0, 3] = 1
            #T[1, 3] = 0.6
            #T[2, 3] = -0.3
            #T[2, 3] = result["trans"][0][0]
            height = frame.shape[0]
            width = frame.shape[1]
            camera_parameters = {
                "intrinsic": K,
                "extrinsic": T,
                "height": height,
                "width": width,
            }
            novel_pose_dataset = avatarmodel.getROMPposeDataset(novel_pose, camera_parameters, height, width)
            #novel_pose_dataset = avatarmodel.getNovelposeDataset()
            novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset,
                                                batch_size = 1,
                                                shuffle = False,
                                                num_workers = 4,)

            #render_path = os.path.join(avatarmodel.model_path, 'novel_pose', "ours_{}".format(epoch))
            #render_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/output/m3c_demo'
            #makedirs(render_path, exist_ok=True)
            
            for idx, batch_data in enumerate(novel_pose_loader):
                batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))

                if model.train_stage ==1:
                    image, mask = avatarmodel.render_free_stage1(batch_data, 59400, 59400)
                    #print("image shape: ", image.shape)
                    if background is not None:
                        mask[mask < 0.5] = 0
                        mask[mask >= 0.5] = 1
                        #print("mask max: ", mask.max())
                        #print("background on")
                        image = image * mask + background * (1 - mask)
                else:
                    image, = avatarmodel.render_free_stage2(batch_data, 59400)

                npimg = image.cpu().numpy()
                gif_save = (np.transpose(npimg, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                '''
                cv2_image = cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_BGR2RGB)
                cv2_save = (cv2_image * 255).clip(0, 255).astype(np.uint8)
                cv2.imshow("Gaussian Avatar", cv2_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                h, w = frame.shape[:2]
                new_h, new_w = int(h * 0.5), int(w * 0.5)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_list.append(frame)
                '''
                gif_list.append(gif_save)
        imageio.mimsave(model.model_path+'/live_noback.gif', gif_list, 'GIF', fps=30, loop=0)
        #imageio.mimsave(model.model_path+'/original_video.gif', video_list, 'GIF', fps=30, loop=0)
                #cv2.imshow("Gaussian Avatar", cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_BGR2RGB))
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break

    cv2.destroyAllWindows()


def show(img):
    plt.clf()
    npimg = img.cpu().numpy()
    #cv2.imshow('render', np.transpose(npimg, (1, 2, 0)))
    #cv2.waitKey(1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show(block=False)
    plt.pause(0.005)
    
def show_cv2(img):
    image = img.detach().cpu().permute(1, 2, 0).numpy()
    image_save = image * 255.0
    image_show = image.astype(np.double)
    cv2.imshow('window', image_show[:, :, ::-1])
        # cv2.waitKey(0)
    #if cv2.waitKey(25) & 0xFF == ord('q'):
        #cv2.destroyAllWindows()
        #break

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--input", default=None, type=str)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)


    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch, args)


