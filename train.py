import os
import torch
import lpips
import torchvision
import open3d as o3d
import sys
import uuid
from tqdm import tqdm
from utils.loss_utils import l1_loss_w, ssim
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams, NetworkParams
from model.avatar_model import AvatarModel
from utils.general_utils import to_cuda, adjust_loss_weights

import kornia
import torch.nn.functional as F
import torchvision.utils as vutils
'''
normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

def compute_normal_loss_from_gt(gt_image, pred_image, predictor):
    """
    使用 GT 圖像生成 normal_gt，並與 pred_image 生成的 normal 比較。
    
    Args:
        gt_image: (B, 3, H, W) tensor in [0,1]
        pred_image: (B, 3, H, W)
        predictor: StableNormal model
    Returns:
        cosine loss between predicted normal and normal_gt
    """
    B = gt_image.size(0)
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    normal_gt_list = []
    normal_pred_list = []

    for b in range(B):
        gt_pil = to_pil(gt_image[b].cpu())
        pred_pil = to_pil(pred_image[b].detach().cpu())

        normal_gt_pil = predictor(gt_pil)
        normal_pred_pil = predictor(pred_pil)

        normal_gt_tensor = to_tensor(normal_gt_pil).to(gt_image.device)
        normal_pred_tensor = to_tensor(normal_pred_pil).to(gt_image.device)

        normal_gt_list.append(normal_gt_tensor)
        normal_pred_list.append(normal_pred_tensor)

    normal_gt = torch.stack(normal_gt_list)      # (B, 3, H, W)
    normal_pred = torch.stack(normal_pred_list)  # (B, 3, H, W)

    # Cosine similarity loss
    cos_sim = F.cosine_similarity(normal_pred, normal_gt, dim=1)  # (B, H, W)
    loss = 1.0 - cos_sim.mean()
    return loss

'''
'''
from ultralytics import YOLO
import numpy as np

yolo_face = YOLO("yolov8n-face.pt")   # 你需要下載對應模型
yolo_hand = YOLO("yolov8n-hand.pt")   # 可以是 community-trained model
yolo_logo = YOLO("your_custom_logo.pt")  # 若無 logo model，可以先略過

def run_yolo_model(model, image, target_classnames):
    """
    單模型 YOLO 預測並回傳 mask
    """
    result = model.predict(image, conf=0.25, verbose=False)[0]
    H, W = image.size[1], image.size[0]
    mask = torch.zeros((H, W))
    
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        name = model.names[int(cls)]
        if name.lower() in target_classnames:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1.0
    return mask.unsqueeze(0)  # (1, H, W)

def get_multi_yolo_mask(image_tensor, device):
    """
    Args:
        image_tensor: (B, 3, H, W)
    Returns:
        masks: (B, 1, H, W)
    """
    B, C, H, W = image_tensor.shape
    all_masks = []
    for b in range(B):
        img_pil = TF.to_pil_image(image_tensor[b].cpu())

        mask_face = run_yolo_model(yolo_face, img_pil, target_classnames=["face"])
        mask_hand = run_yolo_model(yolo_hand, img_pil, target_classnames=["hand"])
        mask_logo = run_yolo_model(yolo_logo, img_pil, target_classnames=["logo"])  # 如果沒有可省略

        combined = torch.maximum(torch.maximum(mask_face, mask_hand), mask_logo)  # union
        all_masks.append(combined.to(device))
    return torch.stack(all_masks).unsqueeze(1)  # (B, 1, H, W)
'''

def compute_edge_loss(pred_img, gt_img):
    # pred_img, gt_img: [B, 3, H, W], normalized to [0,1]
    edge_pred = kornia.filters.Sobel()(pred_img)
    edge_gt = kornia.filters.Sobel()(gt_img)
    loss = torch.nn.functional.l1_loss(edge_pred, edge_gt)
    return loss

def compute_laplacian_loss(pred_img, gt_img):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=torch.float32, device=pred_img.device
    ).view(1, 1, 3, 3)

    def laplacian_map(img):
        channels = []
        for c in range(3):
            channel = img[:, c:c+1, :, :]
            channels.append(F.conv2d(channel, laplacian_kernel, padding=1))
        return torch.cat(channels, dim=1)

    lap_pred = laplacian_map(pred_img)
    lap_gt = laplacian_map(gt_img)

    return F.l1_loss(lap_pred, lap_gt)

def compute_laplacian_map(img):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=torch.float32, device=img.device
    ).view(1, 1, 3, 3)

    def laplacian_channel(c):
        return F.conv2d(c, laplacian_kernel, padding=1)

    channels = []
    for i in range(img.shape[1]):  # loop over RGB channels
        channels.append(laplacian_channel(img[:, i:i+1]))
    return torch.cat(channels, dim=1)

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    import kornia
    return kornia.filters.GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))(x)

def laplacian_pyramid(x, max_levels=3):
    pyramid = []
    current = x
    for _ in range(max_levels):
        blurred = gaussian_blur(current)
        diff = current - blurred
        pyramid.append(diff)
        current = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
    return pyramid

def laplacian_pyramid_loss(pred_img, gt_img, max_levels=3, weight=1.0):
    loss = 0.0
    pred_pyramid = laplacian_pyramid(pred_img, max_levels)
    gt_pyramid = laplacian_pyramid(gt_img, max_levels)
    for l_pred, l_gt in zip(pred_pyramid, gt_pyramid):
        loss += F.l1_loss(l_pred, l_gt)
    return weight * loss








try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def train(model, net, opt, saving_epochs, checkpoint_epochs):
    pose_indices_to_log = [30, 60, 90]  # 自行指定想看的動作索引
    log_interval_epoch = 30  # 每幾個 epoch 存一次

    save_vis_dir = os.path.join(model.model_path, 'log_epoch_vis')
    os.makedirs(save_vis_dir, exist_ok=True)


    tb_writer = prepare_output_and_logger(model, net, opt)
    avatarmodel = AvatarModel(model, net, opt, train=True)
    
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    train_loader = avatarmodel.getTrainDataloader()
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    first_iter = 0
    epoch_start = 0
    data_length = len(train_loader)
    total_iter = opt.epochs * data_length
    avatarmodel.training_setup(total_iteration=total_iter)

    if checkpoint_epochs:
        avatarmodel.load(checkpoint_epochs[0])
        epoch_start += checkpoint_epochs[0]
        first_iter += epoch_start * data_length

    if model.train_stage == 2:
        avatarmodel.stage_load(model.stage1_out_path)
    
    
    progress_bar = tqdm(range(first_iter, total_iter), desc="Training progress")
    ema_loss_for_log = 0.0
    
    for epoch in range(epoch_start + 1, opt.epochs + 1):

        if model.train_stage ==1:
            avatarmodel.net.train()
            avatarmodel.pose.train()
            avatarmodel.transl.train()
            if model.deform_on == True:
                avatarmodel._deformation.train()
            #avatarmodel.attention_net.train()
        else:
            avatarmodel.net.train()
            avatarmodel.pose.eval()
            avatarmodel.transl.eval()
            avatarmodel.pose_encoder.train()
        
        iter_start.record()

        wdecay_rgl = adjust_loss_weights(opt.lambda_rgl, epoch, mode='decay', start=epoch_start, every=20)

        for _, batch_data in enumerate(train_loader):
            first_iter += 1
            #####################rm avatar rectification(deformation model) 2025.04.16##################################
            if model.deform_on == True:
                avatarmodel.update_deform_learning_rate(first_iter)
            ###########################################################################################################
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            gt_image = batch_data['original_image']
            #torchvision.utils.save_image(gt_image, os.path.join(model.model_path, 'log', '{0:05d}_gt'.format(first_iter) + ".png"))

            if model.train_stage ==1:
                if model.deform_on == True:
                    image, points, offset_loss, geo_loss, scale_loss, colors, deform_offset_loss = avatarmodel.train_stage1(batch_data, first_iter, total_iter, epoch)
                    scale_loss = opt.lambda_scale  * scale_loss
                    offset_loss = wdecay_rgl * offset_loss
                    
                    Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                    ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                    loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss + deform_offset_loss * wdecay_rgl
                else:
                    image, points, offset_loss, geo_loss, scale_loss, colors = avatarmodel.train_stage1(batch_data, first_iter, total_iter)
                    scale_loss = opt.lambda_scale  * scale_loss
                    offset_loss = wdecay_rgl * offset_loss
                    
                    Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                    ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
                    #logo_mask = get_multi_yolo_mask(gt_image, device=gt_image.device)
                    #loss_masked = F.l1_loss(image * logo_mask, gt_image * logo_mask)
                    
                    ##################2025.07.06 add edge loss##########################
                    if epoch > 40:
                        loss_edge = compute_edge_loss(image, gt_image)
                        #loss_lap = compute_laplacian_loss(image, gt_image)
                            # ---- 自動細節注意力圖（based on Laplacian）----
                        
                        attention_map = torch.norm(
                            compute_laplacian_map(gt_image),  # shape: (B,1,H,W)
                            dim=1, keepdim=True
                        )
                        attention_map = attention_map / (attention_map.max() + 1e-8)  # normalize to [0,1]


                        #lap_pyr_loss_fn = LaplacianPyramidLoss(max_levels=3, reduction='mean')
                        #lap_pyr_loss = lap_pyr_loss_fn(image, gt_image)
                        lap_pyr_loss = laplacian_pyramid_loss(image, gt_image, max_levels=3, weight=0.01)

                        # ---- Weighted L1 loss ----
                        detail_L1 = F.l1_loss(image * attention_map, gt_image * attention_map)

                        #loss_normal = compute_normal_loss_from_gt(gt_image, image, normal_predictor)
                        


                        loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss + loss_edge * 0.003 + lap_pyr_loss * 0.02  + detail_L1 * 0.005 #+ loss_normal * 0.01#+ loss_masked * 0.05 #loss_lap * 0.01
                    ###############################################################
                    else:
                        loss = scale_loss + offset_loss + Ll1 + ssim_loss + geo_loss #+ detail_L1 * 0.001# + loss_masked * 0.005
            else:
                image, points, pose_loss, offset_loss, colors = avatarmodel.train_stage2(batch_data, first_iter)

                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                loss =  offset_loss + Ll1 + ssim_loss + pose_loss * 10


            if epoch > opt.lpips_start_iter:
                vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                loss = loss + vgg_loss
            
            avatarmodel.zero_grad(epoch)

            loss.backward(retain_graph=True)
            iter_end.record()
            avatarmodel.step(epoch)

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if first_iter % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                #if (first_iter-1) % opt.log_iter == 0:
                    #save_poitns = points.clone().detach().cpu().numpy()
                    #save_colors = colors.clone().detach().cpu().numpy()
                    #for i in range(save_poitns.shape[0]):
                    #    pcd = o3d.geometry.PointCloud()
                     #   pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
                    #    pcd.colors = o3d.utility.Vector3dVector(save_colors)

                     #   o3d.io.write_point_cloud(os.path.join(model.model_path, 'log',f"pred_{int(first_iter)}_{int(i)}.ply") , pcd)

                    #torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', '{0:05d}_pred'.format(first_iter) + ".png"))
                    #torchvision.utils.save_image(gt_image, os.path.join(model.model_path, 'log', '{0:05d}_gt'.format(first_iter) + ".png"))
                '''
                if epoch < 2:
                    pose_indices = batch_data['pose_idx']

                    if pose_indices.numel() == 0:
                        continue

                    i = pose_indices[0].item()

                    heatmap = torch.abs(image - gt_image).mean(dim=1, keepdim=True)
                    heatmap_color = heatmap.repeat(1, 3, 1, 1)

                    vis_grid = torchvision.utils.make_grid(
                        torch.cat([gt_image, image, heatmap_color], dim=0),
                        nrow=3, normalize=True
                    )
                    save_path = os.path.join(save_vis_dir, f"epoch{epoch:03d}_pose{i}.png")
                    torchvision.utils.save_image(vis_grid, save_path)                
                '''
                if epoch > 100 and epoch % log_interval_epoch == 0:
                    pose_indices = batch_data['pose_idx']
                    for pid in pose_indices_to_log:
                        match_idx = (pose_indices == pid).nonzero(as_tuple=True)[0]
                        if match_idx.numel() == 0:
                            continue
                        i = match_idx[0].item()

                        gt = gt_image[i:i+1]
                        pred = image[i:i+1]
                        heatmap = torch.abs(pred - gt).mean(dim=1, keepdim=True)
                        heatmap_color = heatmap.repeat(1, 3, 1, 1)

                        vis_grid = torchvision.utils.make_grid(
                            torch.cat([gt, pred, heatmap_color], dim=0),
                            nrow=3, normalize=True
                        )
                        save_path = os.path.join(save_vis_dir, f"epoch{epoch:03d}_pose{pid}.png")
                        torchvision.utils.save_image(vis_grid, save_path)
                #if first_iter % 500 == 0 and epoch > 50:
                    #vutils.save_image(attention_map, f"{model.model_path}/log/{first_iter:05d}_attn.png", normalize=True)
                #if first_iter % opt.log_iter == 0:
                   # mask_vis = logo_mask.expand(-1, 3, -1, -1)  # (B,3,H,W)
                   # torchvision.utils.save_image(mask_vis, os.path.join(model.model_path, 'log', f"{first_iter:05d}_mask.png"))

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/scale_loss', scale_loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/offset_loss', offset_loss.item(), first_iter)
                # tb_writer.add_scalar('train_loss_patches/aiap_loss', aiap_loss.item(), first_iter)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), first_iter)
                if model.train_stage ==1:
                    tb_writer.add_scalar('train_loss_patches/geo_loss', geo_loss.item(), first_iter)
                else:
                    tb_writer.add_scalar('train_loss_patches/pose_loss', pose_loss.item(), first_iter)
                if epoch > opt.lpips_start_iter:
                    tb_writer.add_scalar('train_loss_patches/vgg_loss', vgg_loss.item(), first_iter)

        if (epoch > saving_epochs[0]) and epoch % model.save_epoch == 0:
            print("\n[Epoch {}] Saving Model".format(epoch))
            avatarmodel.save(epoch)
        if epoch == 200:
            print("\n[Epoch {}] Saving Model".format(epoch))
            avatarmodel.save(epoch)



def prepare_output_and_logger(args, net, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    full_config = {
        "model": vars(args),
        "network": vars(net),
        "optim": vars(opt),
    }
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    os.makedirs(os.path.join(args.model_path, 'log'), exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    import yaml
    with open(os.path.join(args.model_path, "cfg_args.yaml"), 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    np = NetworkParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[150])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(lp.extract(args), np.extract(args), op.extract(args), args.save_epochs, args.checkpoint_epochs)

    print("\nTraining complete.")
