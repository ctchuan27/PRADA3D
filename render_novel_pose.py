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
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image

def rgba_to_p_mode(rgba_img):
    """
    將 RGBA 圖像轉換為 P 模式 (256 色)，並保留透明度
    """
    img = Image.fromarray(rgba_img, 'RGBA')
    # 自動量化顏色，最多 255 種顏色
    p_img = img.convert('P', palette=Image.ADAPTIVE, colors=255)
    # 設定透明索引 (假設左上角為透明背景)
    transparent_index = p_img.getpixel((0, 0))
    p_img.info['transparency'] = transparent_index
    return p_img

def render_sets(model, net, opt, epoch:int):
    with torch.no_grad():
        #background = cv2.imread('./demo/stage.png')
        background = cv2.imread('./demo/rotation/stage.png')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        #background = cv2.resize(background, (1080, 1080), interpolation=cv2.INTER_AREA)
        background = background.transpose(2,0,1)
        background = background / 255.
        background = torch.tensor(background, dtype=torch.float32, device="cuda")
        background = None
        #print("background max: ", background.max())
        #print("background min: ", background.min())
        #model.test_folder=os.getcwd() + '/default_dataset/people_snapshot/m3c_processed/train'
        #model.test_folder=os.getcwd() + '/assets/aist_demo'
        #model.test_folder=os.getcwd() + '/assets/test_pose_v2'
        #model.test_folder=os.getcwd() + '/assets/test_pose'
        model.test_folder=os.getcwd() + '/demo/live'
        avatarmodel = AvatarModel(model, net, opt, train=False, background=None)
        avatarmodel.training_setup()
        avatarmodel.load(epoch)
        #novel_pose_dataset = avatarmodel.getVIBEposeDataset()
        novel_pose_dataset = avatarmodel.getNovelposeDataset()
        #novel_pose_dataset = avatarmodel.getNovelviewDataset()
        novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset, batch_size = 1, shuffle = False, num_workers = 4,)
        #novel_pose_loader = torch.utils.data.DataLoader(novel_view_dataset, batch_size = 1, shuffle = False, num_workers = 4,)

        #render_path = os.path.join(avatarmodel.model_path, 'novel_pose', "ours_{}".format(epoch))
        render_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/output/test_train'
        makedirs(render_path, exist_ok=True)
        
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        #out = cv2.VideoWriter('./render_video.mp4', fourcc, 30.0, (720,1280),False)
        #plt.ion()
        #plt.title("Avatar Rendering")
        gif_list=[]
        frame_paths=[]
        save_dir = os.path.join(model.model_path, "frames_rgba")
        os.makedirs(save_dir, exist_ok=True)
        for idx, batch_data in enumerate(tqdm(novel_pose_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))

            if model.train_stage ==1:
                image, mask = avatarmodel.render_free_stage1(batch_data, 59400, 59400)
                
                ####################gaussians visualize 2025.05.02##############################
                #image, mask = avatarmodel.render_gaussians(batch_data, 59400, 59400)
                ######################################################################
                
                #print("image shape: ", image.shape)
                if background is not None:
                    mask[mask < 0.5] = 0
                    mask[mask >= 0.5] = 1
                    #print("mask max: ", mask.max())
                    image = image * mask + background * (1 - mask)
            else:
                image, = avatarmodel.render_free_stage2(batch_data, 59400)
            #print(image.shape)
            #window_desktop = np.uint8((image.cpu().permute(1, 2, 0).numpy()))
            #window_desktop = cv2.cvtColor(window_desktop, cv2.COLOR_BGR2RGB)
            #print(window_desktop.shape)
            #cv2.namedWindow('window_desktop', cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('window_desktop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            #plt.imshow(window_desktop)
            
            #show(image)
            npimg = image.cpu().numpy()
            gif_save = (np.transpose(npimg, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            h, w = gif_save.shape[:2]
            new_h, new_w = int(h * 0.5), int(w * 0.5)
            gif_save = cv2.resize(gif_save, (new_w, new_h), interpolation=cv2.INTER_AREA)
            '''
            cv2_image = cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2_save = (cv2_image * 255).clip(0, 255).astype(np.uint8)
            cv2.imshow("Gaussian Avatar", cv2_image)
            #cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"), cv2_save)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''
            mask = (mask > 0.5).float()
            alpha = (mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # RGBA frame
            rgba = np.dstack((gif_save, alpha))
            frame_path = os.path.join(save_dir, f"frame_{idx:05d}.png")
            imageio.imwrite(frame_path, rgba)
            frame_paths.append(frame_path)
        '''
        # === 儲存透明背景 WebM ===
        save_path = model.model_path + '/live_noback.webm'
        imageio.mimsave(
            save_path,
            gif_list,       # RGBA 幀列表
            fps=30,
            codec='vp9',    # VP9 支援透明度
            pixelformat='yuva420p'
        )
        '''
            # 合併成 RGBA
            #gif_save = np.dstack((gif_save, alpha))
            #gif_list.append(gif_save)
        #imageio.mimsave(model.model_path+'/live_noback_v4.gif', gif_list, 'GIF', fps=30, loop=0, palettesize=256)  
        
            #npimg = np.uint8(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            #cv2_image = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
            #out.write(cv2_image)
            #cv2.imshow("Image", cv2_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #print(cv2_image.shape)
            
            #torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            #if idx == 0:
            #    image_output = image.unsqueeze(0)
            #else:
                #image = image.unsqueeze(0)
                #image_output = torch.cat((image_output,image), dim=0)
        #out.release()
    os.system(
        f"ffmpeg -y -framerate 30 -i {save_dir}/frame_%05d.png "
        f"-c:v prores_ks -pix_fmt yuva444p10le {model.model_path}/live_noback.mov"
    )
    #cv2.destroyAllWindows()
    #write_video(filename=os.path.join(render_path, "video.mp4"),video_array=torch.FloatTensor(image_output.cpu()),fps=30)

def show(img):
    
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show(block=False)
    plt.pause(0.0001)
    plt.clf()


def set_camera(filename: str, width: int, height: int, 
               fps: float = 20.0, fourcc: str = 'mp4v') -> cv2.VideoWriter:
    return cv2.VideoWriter(
        filename,
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=(width, height)
    )

def write_video():
    render_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/output/test_train'
    batch_images = sorted(Path(render_path).glob('*.png'))
    video_output_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/output/test_train/render_video.mp4'

    camera = None
    for ip in tqdm(batch_images):
        frame = cv2.imread(ip.as_posix())
        
        # process ...
        
        if camera is None:
            h, w, _ = frame.shape
            camera = set_camera(video_output_path, w, h)
        camera.write(frame)
    camera.release()


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)


    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch,)

    #write_video()

    
