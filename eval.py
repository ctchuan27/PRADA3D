import torch
import os
from tqdm import tqdm
from os import makedirs
import torch.nn as nn
import torchvision
import numpy as np
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import  AvatarModel
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
from utils.general_utils import to_cuda, adjust_loss_weights
from utils.loss_utils import l1_loss_w, ssim
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from torch.cuda.amp import custom_fwd
class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        #self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        # rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        # rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


def render_sets(model, net, opt, epoch:int):

    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup(total_iteration=17100)
        avatarmodel.load(epoch, test=False)
        
        test_dataset = avatarmodel.getTestDataset()
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 4,)

        data_length = len(test_loader)
        total_iter = opt.epochs * data_length

        render_path = os.path.join(avatarmodel.model_path, 'test_free', "ours_{}".format(epoch))
        gt_path  = os.path.join(avatarmodel.model_path, 'test_free', 'gt_image')

        makedirs(render_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        results = []

        for idx, batch_data in enumerate(tqdm(test_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            if model.sr is True:
                print("Evaluation for super-resolution")
                gt_image = batch_data['eval_image']
            else:
                gt_image = batch_data['original_image']

            if model.train_stage == 1:
                image, mask = avatarmodel.render_free_stage1(batch_data, 59400, 59400)
            else:
                image, = avatarmodel.render_free_stage2(batch_data, 59400)

            if model.sr is True:
                transform = T.Resize((gt_image.shape[2], gt_image.shape[3]))
                image = transform(image).unsqueeze(0)
            else:
                if model.downscale_eval == True:
                    transform = T.Resize((int(gt_image.shape[2]/2), int(gt_image.shape[3]/2)))
                    gt_image = transform(gt_image)
                    image = transform(image)
                image = image.unsqueeze(0)
            results.append(evaluator(image, gt_image))
              
            torchvision.utils.save_image(gt_image, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        with open("results.txt", "w") as f:
            psnr = torch.stack([r['psnr'] for r in results]).mean().item()
            print(f"PSNR: {psnr:.2f}")
            f.write(f"PSNR: {psnr:.2f}\n")

            ssim = torch.stack([r['ssim'] for r in results]).mean().item()
            print(f"SSIM: {ssim:.4f}")
            f.write(f"SSIM: {ssim:.4f}\n")

            lpips = torch.stack([r['lpips'] for r in results]).mean().item()
            print(f"LPIPS: {lpips:.4f}")
            f.write(f"LPIPS: {lpips:.4f}\n")        

        print('save video...')


###################################2025.04.12 combined GART's test time optimization####################################
'''
def testtime_pose_optimization(
        model, net, opt, epoch:int,
        pose_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        As=None,
        pose_As_lr=1e-3,
    ):
    torch.cuda.empty_cache()
    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()
    avatarmodel = AvatarModel(model, net, opt, train=False)
    avatarmodel.training_setup()
    avatarmodel.load(epoch, test=False)
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    
    test_dataset = avatarmodel.getTestDataset()
    avatarmodel.eval_setup(model, test_dataset, batch_size=1, pose_lr=pose_lr, decay_steps=decay_steps, decay_factor=decay_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 1,)

    #render_path = os.path.join(avatarmodel.model_path, 'test_free', "ours_{}".format(epoch))
    #gt_path  = os.path.join(avatarmodel.model_path, 'test_free', 'gt_image')

    #makedirs(render_path, exist_ok=True)
    #makedirs(gt_path, exist_ok=True)
    #results = []

    for inner_step in tqdm(range(steps), desc="test time optimization progress"):
        avatarmodel.net.eval()
        avatarmodel.pose.eval()
        avatarmodel.transl.eval()
        avatarmodel.eval_pose.train()
        avatarmodel.eval_transl.train()
        #avatarmodel.attention_net.eval()
        
        #wdecay_rgl = adjust_loss_weights(opt.lambda_rgl, epoch, mode='decay', start=epoch_start, every=20)
        avatarmodel.eval_zero_grad()
        total_loss = 0.0
        for idx, batch_data in enumerate(test_loader):
            
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']

            #if model.train_stage == 1:
                #image, _, offset_loss, geo_loss, scale_loss = avatarmodel.eval_stage1(batch_data, 59400)
            #else:
                #image, = avatarmodel.render_free_stage2(batch_data, 59400)

            image, _, offset_loss, geo_loss, scale_loss = avatarmodel.eval_testtime_optimization(batch_data, 59400)

            #scale_loss = opt.lambda_scale  * scale_loss
            #offset_loss = wdecay_rgl * offset_loss
            Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
            ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
            #loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss
            loss = Ll1 + ssim_loss

            vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
            loss = loss + vgg_loss
            total_loss += loss

        total_loss = total_loss / len(test_loader)
        total_loss.backward()
        avatarmodel.eval_step()

            #results.append(evaluator(image.unsqueeze(0), gt_image))


    ########################start inference############################   
    avatarmodel.net.eval()
    avatarmodel.pose.eval()
    avatarmodel.transl.eval()
    avatarmodel.eval_pose.eval()
    avatarmodel.eval_transl.eval()

    with torch.no_grad():
        render_path = os.path.join(avatarmodel.model_path, 'test_free', "ours_{}".format(epoch))
        gt_path  = os.path.join(avatarmodel.model_path, 'test_free', 'gt_image')

        makedirs(render_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        results = []

        for idx, batch_data in enumerate(tqdm(test_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']

            image, mask = avatarmodel.eval_stage1(batch_data, 59400)

            results.append(evaluator(image.unsqueeze(0), gt_image))
              
            torchvision.utils.save_image(gt_image, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        with open("results.txt", "w") as f:
            psnr_loss = torch.stack([r['psnr'] for r in results]).mean().item()
            print(f"PSNR: {psnr_loss:.2f}")
            f.write(f"PSNR: {psnr_loss:.2f}\n")

            ssim_loss = torch.stack([r['ssim'] for r in results]).mean().item()
            print(f"SSIM: {ssim_loss:.4f}")
            f.write(f"SSIM: {ssim_loss:.4f}\n")

            lpips_loss = torch.stack([r['lpips'] for r in results]).mean().item()
            print(f"LPIPS: {lpips_loss:.4f}")
            f.write(f"LPIPS: {lpips_loss:.4f}\n")        

        print('save video...') 
'''


def testtime_pose_optimization(
        model, net, opt, epoch:int,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        As=None,
        pose_As_lr=1e-3,
    ):

    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()
    avatarmodel = AvatarModel(model, net, opt, train=False)
    avatarmodel.training_setup()
    avatarmodel.load(epoch, test=False)

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()

    test_dataset = avatarmodel.getTestDataset()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 1,)

    avatarmodel.net.eval()
    if model.deform_on == True:
        avatarmodel._deformation.eval()
    avatarmodel.pose.eval()
    avatarmodel.transl.eval()

    render_path = os.path.join(avatarmodel.model_path, 'test_free', "ours_{}".format(epoch))
    gt_path  = os.path.join(avatarmodel.model_path, 'test_free', 'gt_image')

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    results = []

    for idx, batch_data in enumerate(tqdm(test_loader, desc="test time optimization progress")):
        batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
        new_pose_b, new_pose_r, new_trans = avatarmodel.eval_testtime_optimization(
            batch_data, 59400, evaluator, opt, loss_fn_vgg,
            pose_base_lr=pose_base_lr,
            pose_rest_lr=pose_rest_lr,
            trans_lr=trans_lr,
            steps=steps,
            decay_steps=decay_steps,
            decay_factor=decay_factor,
            As=None,
            pose_As_lr=pose_As_lr,
        )
        pose = torch.cat([new_pose_b, new_pose_r], dim=1).detach()
        trans = new_trans.detach()
    
        with torch.no_grad():
            gt_image = batch_data['original_image']
            image, mask = avatarmodel.eval_stage1(batch_data, 59400, pose, trans)

            results.append(evaluator(image.unsqueeze(0), gt_image))
              
            torchvision.utils.save_image(gt_image, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    with open("results.txt", "w") as f:
        psnr_loss = torch.stack([r['psnr'] for r in results]).mean().item()
        print(f"PSNR: {psnr_loss:.2f}")
        f.write(f"PSNR: {psnr_loss:.2f}\n")

        ssim_loss = torch.stack([r['ssim'] for r in results]).mean().item()
        print(f"SSIM: {ssim_loss:.4f}")
        f.write(f"SSIM: {ssim_loss:.4f}\n")

        lpips_loss = torch.stack([r['lpips'] for r in results]).mean().item()
        print(f"LPIPS: {lpips_loss:.4f}")
        f.write(f"LPIPS: {lpips_loss:.4f}\n")        

    print('save video...') 

####################################################################################################################################




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--tto", action="store_true")
    parser.add_argument("--tto_steps", default=300, type=int)
    #parser.add_argument("--sr", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    if args.tto is True:
        testtime_pose_optimization(model.extract(args), network.extract(args), op.extract(args), args.epoch, pose_base_lr=3e-6, pose_rest_lr=3e-6, trans_lr=3e-6, steps=args.tto_steps, decay_steps=70,decay_factor=0.5)
    else:
        render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch,)
