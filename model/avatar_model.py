import torch
import numpy as np
import torch
import os
import numpy as np
import torch.nn as nn
from submodules import smplx
import trimesh
from scene.dataset_mono import MonoDataset_train, MonoDataset_test, MonoDataset_novel_pose, MonoDataset_novel_view, MonoDataset_novel_pose_VIBE, ROMP_novel_pose_webcam
from utils.general_utils import worker_init_fn
from utils.system_utils import mkdir_p
from model.network import POP_no_unet
from utils.general_utils import load_masks
from gaussian_renderer import render_batch, render_batch_custom_background
from os.path import join
import torch.nn as nn
from model.modules  import UnetNoCond5DS
from model.network import POP_no_unet
####################ram_networks criss cross attention######################
from model.ram_networks.styleunet.styleunet import TriPlane_Conv
############################################################################
###################test time optimization######################
from tqdm import tqdm
import lpips
from utils.loss_utils import l1_loss_w, ssim
###############################################################
#####################rm avatar rectification(deformation model) 2025.04.16##################################
from model.rmavatar_networks.scene.deformation import deform_network
#from model.deformnet import deform_network
from model.rmavatar_networks.utils.general_utils import get_expon_lr_func
from torch.utils.data import Dataset
###########################################################################################
import math

class RMAvatar_point_dataset(Dataset):
    # data loading
    def __init__(self, point, scale, rotation):
        self.point = point
        self.scale = scale
        self.rotation = rotation
    # working for indexing
    def __getitem__(self, index):
        
        return self.point[index], self.scale[index], self.rotation[index]

    # return the length of our dataset
    def __len__(self):
        
        return self.point.shape[0]





class AvatarModel:
    def __init__(self, model_parms, net_parms, opt_parms, load_iteration=None, train=True, background=None):

        self.model_parms = model_parms
        self.net_parms = net_parms
        self.opt_parms = opt_parms
        self.model_path = model_parms.model_path
        self.loaded_iter = None
        self.train = train
        self.train_mode = model_parms.train_mode
        self.gender = self.model_parms.smpl_gender
        self.deform_on = self.model_parms.deform_on

        if train:
            self.batch_size = self.model_parms.batch_size
        else:
            self.batch_size = 1

        if train:
            split = 'train'
        else:
            split = 'test'

        self.train_dataset  = MonoDataset_train(model_parms)
        self.smpl_data = self.train_dataset.smpl_data

        # partial code derive from POP (https://github.com/qianlim/POP)
        assert model_parms.smpl_type in ['smplx', 'smpl']
        if model_parms.smpl_type == 'smplx':
            self.smpl_model = smplx.SMPLX(model_path=self.model_parms.smplx_model_path, gender = self.gender, use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smplx')
            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smplx_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path, split, 'smplx_cano_joint_mat.pth')
            joint_num = 55
        
        else:
            #print("self.model_parms.inp_posmap_size: ", self.model_parms.inp_posmap_size)
            self.smpl_model = smplx.SMPL(model_path=self.model_parms.smpl_model_path, gender = self.gender, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smpl')

            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smpl_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path,  split, 'smpl_cano_joint_mat.pth')
            joint_num = 24

        self.uv_coord_map = uv_coord_map
        self.valid_idx = valid_idx

        if model_parms.fixed_inp:
            fix_inp_map = torch.from_numpy(np.load(inp_map_path)['posmap' + str(self.model_parms.inp_posmap_size)].transpose(2,0,1)).cuda()
            self.fix_inp_map = fix_inp_map[None].expand(self.batch_size, -1, -1, -1)
        
        ## query_map store the sampled points from the cannonical smpl mesh, shape as [512. 512, 3] 
        #print('query_map_path: ', query_map_path)
        query_map = torch.from_numpy(np.load(query_map_path)['posmap' + str(self.model_parms.query_posmap_size)]).reshape(-1,3)
        query_points = query_map[valid_idx.cpu(), :].cuda().contiguous()
        
        self.query_points = query_points[None].expand(self.batch_size, -1, -1)
        
        # we fix the opacity and rots of 3d gs as described in paper 
        self.fix_opacity = torch.ones((self.query_points.shape[1], 1)).cuda()
        rots = torch.zeros((self.query_points.shape[1], 4), device="cuda")
        rots[:, 0] = 1
        self.fix_rotation = rots
        
        # we save the skinning weights from the cannonical mesh
        query_lbs = torch.from_numpy(np.load(query_lbs_path)).reshape(self.model_parms.query_posmap_size*self.model_parms.query_posmap_size, joint_num)
        self.query_lbs = query_lbs[valid_idx.cpu(), :][None].expand(self.batch_size, -1, -1).cuda().contiguous()
        
        self.inv_mats = torch.linalg.inv(torch.load(mat_path)).expand(self.batch_size, -1, -1, -1).cuda()
        print('inv_mat shape: ', self.inv_mats.shape)

        num_training_frames = len(self.train_dataset)
        param = []

        if not torch.is_tensor(self.smpl_data['beta']):
            self.betas = torch.from_numpy(self.smpl_data['beta'][0])[None].expand(self.batch_size, -1).cuda()
        else:
            self.betas = self.smpl_data['beta'][0][None].expand(self.batch_size, -1).cuda()

        if model_parms.smpl_type == 'smplx':
            self.pose = torch.nn.Embedding(num_training_frames, 66, _weight=self.train_dataset.pose_data, sparse=True).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
            param += list(self.transl.parameters())
        else:
            self.pose = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data, sparse=True).cuda()
            param += list(self.pose.parameters())

            self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
            param += list(self.transl.parameters())
        
        self.optimizer_pose = torch.optim.SparseAdam(param, 1.0e-4)
        
        if background is not None:
            self.background = torch.tensor(background, dtype=torch.float32, device="cuda")
        else:
            bg_color = [1, 1, 1] if model_parms.white_background else [0, 0, 0]
            #bg_color = [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.rotation_activation = torch.nn.functional.normalize
        self.sigmoid_activation =  nn.Sigmoid()
        
        self.net_set(self.model_parms.train_stage)

    def net_set(self, mode):
        assert mode in [0, 1, 2]
        #self.attention_net = TriPlane_Conv(inp_ch=self.net_parms.c_geom, hid_channel=self.net_parms.c_geom*2, out_channel=self.net_parms.c_geom).cuda()
        self.net = POP_no_unet(
            c_geom=self.net_parms.c_geom, # channels of the geometric features
            #c_geom=30,
            geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            nf=self.net_parms.nf, # num filters for the unet
            #nf=30,
            hsize=self.net_parms.hsize, # hidden layer size of the ShapeDecoder MLP
            up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            uv_feat_dim=2, # input dimension of the uv coordinates
        ).cuda()
        #print("self.model_parms.inp_posmap_size: ", self.model_parms.inp_posmap_size)
        '''
        sapiens_feat_path = os.listdir(join(self.model_parms.source_path, "train/sapiens_2b"))
        sapiens_feature = np.zeros((1920, 64, 64))
        #sapiens_features = []
        for f in sapiens_feat_path:
            sapiens_feature = sapiens_feature + np.load(join(self.model_parms.source_path, "train/sapiens_2b",f))
            #sapiens_features.append(torch.tensor(np.load(join(self.model_parms.source_path, "train/sapiens_2b",f))).float().cuda())
        sapiens_feature = sapiens_feature / len(sapiens_feat_path)
        sapiens_feature = sapiens_feature / max(sapiens_feature.max(), abs(sapiens_feature.min())) #*0.010
        sapiens_feature = torch.tensor(sapiens_feature).float().cuda()
        self.sapiens_feature = sapiens_feature.reshape(1, 120, 256, 256)
        #self.sapiens_feature = sapiens_features
        '''
        #self.sapiens_feature = None
        #feature = torch.clamp(feature, -0.55, 0.55)
        #print("feature max min: ", feature.max(),feature.min())
        geo_feature = torch.ones(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        self.geo_feature = nn.Parameter(geo_feature.requires_grad_(True))
        #self.geo_feature = nn.Parameter(self.sapiens_feature.requires_grad_(True))
        #print("geo_feature shape: ", self.geo_feature.shape)
        #print("geo_feature max min: ", self.geo_feature.max(),self.geo_feature.min())

        
        if self.model_parms.train_stage == 2:
            self.pose_encoder = UnetNoCond5DS(
                input_nc=3,
                output_nc=self.net_parms.c_pose,
                nf=self.net_parms.nf,
                up_mode=self.net_parms.up_mode,
                use_dropout=False,
            ).cuda()
        
        #####################rm avatar rectification(deformation model) 2025.04.16##################################
        if self.deform_on == True:
            #self._deformation = deform_network(points_channel=3, poses_channel=72).cuda()
            self._deformation = deform_network(self.model_parms).cuda()

        ###########################################################################################################

    def training_setup(self, total_iteration=None):
        if self.model_parms.train_stage  ==1:
            self.optimizer = torch.optim.Adam(
            [
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net},
                {"params": self.geo_feature, "lr": self.opt_parms.lr_geomfeat},
                #{"params": self.attention_net.parameters(), "lr": self.opt_parms.lr_net}
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)
            #####################rm avatar rectification(deformation model) 2025.04.16##################################
            if self.deform_on == True:
                l = []
                self.deform_schedulers = []
                self.deform_lr_schedulers = {}
                
                print(f'[RMAvatarOptim] optim_deformation, lr={self.opt_parms.deform_lr}')
                l.append({
                    'params': list(self._deformation.get_mlp_parameters()),
                    'lr': self.opt_parms.deform_lr,
                    'name': '_deformation'
                })
                self.deform_lr_schedulers['_deformation'] = get_expon_lr_func(
                            lr_init=self.opt_parms.deform_sched_lr_init,
                            lr_final=self.opt_parms.deform_sched_lr_final,
                            lr_delay_mult=self.opt_parms.deform_sched_lr_delay_mult,
                            max_steps=total_iteration)

                print(f'[RMAvatarOptim] optim_grid, lr={self.opt_parms.grid_lr}')
                l.append({
                    'params': list(self._deformation.get_grid_parameters()),
                    'lr': self.opt_parms.grid_lr,
                    'name': '_grid'
                })
                self.deform_lr_schedulers['_grid'] = get_expon_lr_func(
                            lr_init=self.opt_parms.grid_sched_lr_init,
                            lr_final=self.opt_parms.grid_sched_lr_final,
                            lr_delay_mult=self.opt_parms.grid_sched_lr_delay_mult,
                            max_steps=total_iteration)
                
                self.deform_optimizer = torch.optim.Adam(l, lr=5e-4, eps=1e-15)
                '''
                self.deform_optimizer = torch.optim.Adam([
                    {"params": self._deformation.parameters(), "lr": self.opt_parms.deform_lr, 'name': '_deformation'},
                ], lr=5e-4, eps=1e-15)
                self.deform_lr_schedulers['_deformation'] = get_expon_lr_func(
                            lr_init=self.opt_parms.deform_sched_lr_init,
                            lr_final=self.opt_parms.deform_sched_lr_final,
                            lr_delay_mult=self.opt_parms.deform_sched_lr_delay_mult,
                            max_steps=total_iteration)
                '''
                deform_milestones = 10000
                deform_milestones = [i for i in range(1, total_iteration) if i % deform_milestones == 0]
                deform_decay = 0.33
                self.deform_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                    self.deform_optimizer,
                    milestones=deform_milestones,
                    gamma=deform_decay,
                ))
            #############################################################################################################

        else:
            self.optimizer = torch.optim.Adam(
            [   
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net * 0.1},
                {"params": self.pose_encoder.parameters(), "lr": self.opt_parms.lr_net},
            ])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)

    '''
    def eval_setup(self, model_parms,eval_dataset, batch_size=1, pose_lr=3e-3, decay_steps=30, decay_factor=0.5):
        
        smpl_data_eval = eval_dataset.smpl_data

        num_eval_frames = len(eval_dataset)
        param = []

        if not torch.is_tensor(smpl_data_eval['beta']):
            self.eval_betas = torch.from_numpy(smpl_data_eval['beta'][0])[None].expand(batch_size, -1).cuda()
        else:
            self.eval_betas = smpl_data_eval['beta'][0][None].expand(batch_size, -1).cuda()

        if model_parms.smpl_type == 'smplx':
            self.eval_pose = torch.nn.Embedding(num_eval_frames, 66, _weight=eval_dataset.pose_data, sparse=True).cuda()
            param += list(self.eval_pose.parameters())

            self.eval_transl = torch.nn.Embedding(num_eval_frames, 3, _weight=eval_dataset.transl_data, sparse=True).cuda()
            param += list(self.eval_transl.parameters())
        else:
            self.eval_pose = torch.nn.Embedding(num_eval_frames, 72, _weight=eval_dataset.pose_data, sparse=True).cuda()
            param += list(self.eval_pose.parameters())

            self.eval_transl = torch.nn.Embedding(num_eval_frames, 3, _weight=eval_dataset.transl_data, sparse=True).cuda()
            param += list(self.eval_transl.parameters())
        
        self.optimizer_pose_eval = torch.optim.SGD(param, pose_lr)
        self.scheduler_eval = torch.optim.lr_scheduler.StepLR(self.optimizer_pose_eval, step_size=decay_steps, gamma=decay_factor)
    '''
    #####################rm avatar rectification(deformation model) 2025.04.16##################################
    def update_deform_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.deform_optimizer.param_groups:
            if param_group['name'] in self.deform_lr_schedulers:
                lr = self.deform_lr_schedulers[param_group['name']](iteration)
                param_group['lr'] = lr
        return lr
    #############################################################################################################

    def save(self, iteration):
        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(iteration))
        mkdir_p(net_save_path)
        if self.model_parms.train_stage  == 1:
            if self.deform_on == True:
                torch.save(
                    {
                    ######################################################
                    #"sapiens_feature": self.sapiens_feature,
                    #"attention_net": self.attention_net.state_dict(),
                    ######################################################
                    "deform_optimizer": self.deform_optimizer.state_dict(),
                    "deform_schedulers": self.deform_schedulers,
                    "_deformation": self._deformation.state_dict(),
                    ######################################################
                    "net": self.net.state_dict(),
                    "geo_feature": self.geo_feature,
                    "pose": self.pose.state_dict(),
                    "transl": self.transl.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()}, 
                os.path.join(net_save_path,  "net.pth"))
            else:
                torch.save(
                    {
                    ######################################################
                    #"sapiens_feature": self.sapiens_feature,
                    #"attention_net": self.attention_net.state_dict(),
                    ######################################################
                    "net": self.net.state_dict(),
                    "geo_feature": self.geo_feature,
                    "pose": self.pose.state_dict(),
                    "transl": self.transl.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()}, 
                os.path.join(net_save_path,  "net.pth"))
        else:
            torch.save(
                {
                "pose_encoder": self.pose_encoder.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}, 
            os.path.join(net_save_path,  "pose_encoder.pth"))

    def load(self, iteration, test=False):

        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(iteration))

        if self.model_parms.train_stage  ==1:
            saved_model_state = torch.load(
                os.path.join(net_save_path, "net.pth"))
            print('load pth: ', os.path.join(net_save_path, "net.pth"))
            self.net.load_state_dict(saved_model_state["net"], strict=False)
            if self.deform_on == True:
                self._deformation.load_state_dict(saved_model_state["_deformation"], strict=False)
            #self.attention_net.load_state_dict(saved_model_state["attention_net"], strict=False)
        
        if self.model_parms.train_stage  ==2:
            saved_model_state = torch.load(
                os.path.join(net_save_path, "pose_encoder.pth"))
            print('load pth: ', os.path.join(net_save_path, "pose_encoder.pth"))
            self.net.load_state_dict(saved_model_state["net"], strict=False)

        if self.model_parms.train_stage  ==1:
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            # if self.train_mode == 0:
            #print("self.geo_feature shape: ", self.geo_feature.shape)
            #print("saved_model_state['geo_feature'].shape: ", saved_model_state["geo_feature"].squeeze(0).shape)
            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
            #self.sapiens_feature.data[...] = saved_model_state["sapiens_feature"].data[...]
            #atten_feat = saved_model_state["geo_feature"].data[...]
            #self.geo_feature.data[...] = atten_feat.squeeze(0)

        if self.model_parms.train_stage  ==2:
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            # if self.train_mode == 0:

            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]

            self.pose_encoder.load_state_dict(saved_model_state["pose_encoder"], strict=False)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved_model_state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved_model_state["scheduler"])
        if self.deform_on == True:
            if self.deform_optimizer is not None:
                self.deform_optimizer.load_state_dict(saved_model_state["deform_optimizer"])
            #if self.deform_schedulers is not None:
                #self.deform_schedulers.load_state_dict(saved_model_state["deform_schedulers"])


    def stage_load(self, ckpt_path):

        net_save_path = ckpt_path
        print('load pth: ', os.path.join(net_save_path, "net.pth"))
        saved_model_state = torch.load(
            os.path.join(net_save_path, "net.pth"))
        
        self.net.load_state_dict(saved_model_state["net"], strict=False)
        self.pose.load_state_dict(saved_model_state["pose"], strict=False)
        self.transl.load_state_dict(saved_model_state["transl"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]

    def stage2_load(self, epoch):
    
        pose_encoder_path = os.path.join(self.model_parms.project_path, self.model_path, "net/iteration_{}".format(epoch))

        pose_encoder_state = torch.load(
            os.path.join(pose_encoder_path, "pose_encoder.pth"))
        print('load pth: ', os.path.join(pose_encoder_path, "pose_encoder.pth"))

        self.net.load_state_dict(pose_encoder_state["net"], strict=False)
        self.pose.load_state_dict(pose_encoder_state["pose"], strict=False)
        self.transl.load_state_dict(pose_encoder_state["transl"], strict=False)
        # if self.train_mode == 0:
        self.geo_feature.data[...] = pose_encoder_state["geo_feature"].data[...]
        self.pose_encoder.load_state_dict(pose_encoder_state["pose_encoder"], strict=False)

    def getTrainDataloader(self,):
        return torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            num_workers = 4,
                                            worker_init_fn = worker_init_fn,
                                            drop_last = True)

    def getTestDataset(self,):
        self.test_dataset = MonoDataset_test(self.model_parms)
        return self.test_dataset
    
    def getNovelposeDataset(self,):
        self.novel_pose_dataset = MonoDataset_novel_pose(self.model_parms)
        return self.novel_pose_dataset
        
    def getVIBEposeDataset(self,):
        self.novel_pose_dataset = MonoDataset_novel_pose_VIBE(self.model_parms)
        return self.novel_pose_dataset
    
    def getROMPposeDataset(self, novel_pose, camera_parameters, height, width):
        self.novel_pose_dataset = ROMP_novel_pose_webcam(self.model_parms, novel_pose, camera_parameters,  height, width)
        return self.novel_pose_dataset

    def getNovelviewDataset(self,):
        self.novel_view_dataset = MonoDataset_novel_view(self.model_parms)
        return self.novel_view_dataset

    def zero_grad(self, epoch):
        self.optimizer.zero_grad()

        if self.model_parms.train_stage  ==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                self.optimizer_pose.zero_grad()
        
        if self.deform_on == True:
            if epoch > self.opt_parms.deform_start_iter:
                self.deform_optimizer.zero_grad(set_to_none=True)

    '''
    def eval_zero_grad(self):
        self.optimizer_pose_eval.zero_grad()
    '''

    def step(self, epoch):
        self.optimizer.step()
        self.scheduler.step()
        if self.model_parms.train_stage==1:
            if epoch > self.opt_parms.pose_op_start_iter:
                self.optimizer_pose.step()
        
        if self.deform_on == True:
            if epoch > self.opt_parms.deform_start_iter:
                self.deform_optimizer.step()
                for scheduler in self.deform_schedulers:
                    scheduler.step()
        
    '''
    def eval_step(self):
        self.optimizer_pose_eval.step()
    '''

    def train_stage1(self, batch_data, iteration, total_iteration=None, epoch=None):
        
        rendered_images = []
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)
        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        
        #print("self.geo_feature shape: ", self.geo_feature.shape)
        #print("self.geo_feature: ", self.geo_feature)
        #print("self.uv_coord_map shape: ", self.uv_coord_map.shape)


        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        #sapiens_feature = self.sapiens_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        #print("geom_featmap expand shape: ", geom_featmap.shape)
        #print("uv_coord_map expand shape: ", uv_coord_map.shape)
        #attention_featmap = self.attention_net.forward(geom_featmap, sapiens_feature)
        #self.attention_featmap = attention_featmap.squeeze(0)
        #print("geom_feamap: ",geom_feamap.shape)
        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    #geom_featmap=attention_featmap,
                                                    uv_loc=uv_coord_map,
                                                    sapiens_feature=None)

        
        #print("pred_res shape: ", pred_res.shape)    
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()
        #print("pred_point_res shape: ", pred_point_res.shape)

        cano_deform_point = pred_point_res + self.query_points
        #print("cano_deform_point shape: ", cano_deform_point.shape)

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        geo_loss = torch.mean(self.geo_feature**2)
        scale_loss = torch.mean(pred_scales)

        #####################rm avatar rectification(deformation model) 2025.04.16##################################
        rotations_batch = self.fix_rotation.expand(self.batch_size, -1, -1).contiguous()
        #if self.deform_on == True:
        #    full_pred, pred_scales, rotations_batch, offset = self._deformation(full_pred, pred_scales, rotations_batch, pose_batch, iteration, total_iteration)
        #    deform_offset_loss = torch.mean(offset ** 2)
        ############################################################################################################
        deform_offset_loss = 0.0
        #if self.deform_on == True:
        #    full_pred, pred_scales, rotations_batch, offset = self._deformation(full_pred, pred_scales, rotations_batch, pose_batch)
        #    deform_offset_loss = torch.mean(offset ** 2)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]

            #####################rm avatar rectification(deformation model) 2025.04.16##################################
            rotations = rotations_batch[batch_index]
            pose = pose_batch[batch_index][3:]

            #points_dict = {}
            #points_dict['points'] = points
            #points_dict['scales'] = scales
            #points_dict['rotations'] = rotations
            
            if self.deform_on == True:
                '''
                RM_dataset = RMAvatar_point_dataset(points, scales, rotations)
                RM_dataloader = torch.utils.data.DataLoader(RM_dataset, batch_size=100, shuffle=False)
                total_offset = []
                idx = 0
                for point, scale, rotation in RM_dataloader:
                    point, scale, rotation, offset = self._deformation(point, scale, rotation, pose, iteration, total_iteration)
                    if idx == 0:
                        total_offset = offset
                        points = point
                        scales = scale
                        rotations = rotation
                        idx = 1
                    else:
                        total_offset = torch.cat((total_offset,offset), dim=0)
                        points = torch.cat((points, point), dim=0)
                        scales = torch.cat((scales, scale), dim=0)
                        rotations = torch.cat((rotations, rotation), dim=0)
                    #deform_offset_loss = deform_offset_loss + torch.mean(offset ** 2) #/ len(RM_dataloader)
                deform_offset_loss = deform_offset_loss + torch.mean(total_offset) ** 2
                '''
                if epoch > self.opt_parms.deform_start_iter:
                    points, scales, rotations, offset = self._deformation(points, scales, rotations, pose, iteration, total_iteration)
                    deform_offset_loss = deform_offset_loss + torch.mean(offset) ** 2
            ############################################################################################################
            #print("pose shape: ", pose.shape)
            #print("points shape: ", points.shape)
            #print("scales shape: ", scales.shape)

            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    #rotations=self.fix_rotation,
                    rotations=rotations,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )
        
        if self.deform_on == True:
            #deform_offset_loss = deform_offset_loss / self.batch_size
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, scale_loss, colors, deform_offset_loss
        else:
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, scale_loss, colors

    def train_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        #print("pred_res shape: ", pred_res.shape)
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()
        #print("pred_point_res shape: ", pred_point_res.shape)
        cano_deform_point = pred_point_res + self.query_points
        #print("cano_deform_point shape: ", cano_deform_point.shape)
        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        pose_loss = torch.mean(pose_featmap ** 2)
        scale_loss = torch.mean(pred_scales)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]




            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0), full_pred, pose_loss, offset_loss, colors
        
###########################2025.04.13 evaluation##########################
    def eval_testtime_optimization(self, batch_data, iteration, evaluator, opt, loss_fn_vgg,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        As=None,
        pose_As_lr=1e-3,
    ):
        
        torch.cuda.empty_cache()
        self.geo_feature.requires_grad_(False)
        
        evaluator.eval()
        self.net.eval()
        if self.deform_on == True:
            self._deformation.eval()
        self.pose.eval()
        self.transl.eval()

        pose_b = batch_data['pose_data'][:, :3]
        if self.model_parms.smpl_type == 'smplx':
            pose_r = batch_data['pose_data'][:, 3:66]
        else:    
            pose_r = batch_data['pose_data'][:, 3:]
        trans = batch_data['transl_data']

        pose_b = pose_b.detach().clone()
        pose_r = pose_r.detach().clone()
        trans = trans.detach().clone()
        pose_b.requires_grad_(True)
        pose_r.requires_grad_(True)
        trans.requires_grad_(True)

        gt_image = batch_data['original_image']

        optim_l = [
            {"params": [pose_b], "lr": pose_base_lr},
            {"params": [pose_r], "lr": pose_rest_lr},
            {"params": [trans], "lr": trans_lr},
        ]
        optimizer_smpl = torch.optim.SGD(optim_l)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_smpl, step_size=decay_steps, gamma=decay_factor
        )

        for inner_step in range(steps):
            # optimize
            rendered_images = []
            accumulations = []
            optimizer_smpl.zero_grad()
            if self.model_parms.smpl_type == 'smplx':
                rest_pose = batch_data['rest_pose']
                live_smpl = self.smpl_model.forward(betas = self.betas,
                                                    global_orient = pose_b,
                                                    transl = trans,
                                                    body_pose = pose_r,
                                                    jaw_pose = rest_pose[:, :3],
                                                    leye_pose=rest_pose[:, 3:6],
                                                    reye_pose=rest_pose[:, 6:9],
                                                    left_hand_pose= rest_pose[:, 9:54],
                                                    right_hand_pose= rest_pose[:, 54:])
            else:
                live_smpl = self.smpl_model.forward(betas=self.betas,
                                    global_orient=pose_b,
                                    transl = trans,
                                    body_pose=pose_r)
            
            cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
            geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
            uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
            #sapiens_feature = self.sapiens_feature.expand(self.batch_size, -1, -1, -1).contiguous()
            #attention_featmap = self.attention_net.forward(geom_featmap, sapiens_feature)
            #print("geom_featmap max min: ", geom_featmap.max(),geom_featmap.min())
            #print("uv_coord_map max min: ", uv_coord_map.max(),uv_coord_map.min())
            #print("sapiens_feature max min: ", sapiens_feature.max(),sapiens_feature.min())
            pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                        geom_featmap=geom_featmap,
                                                        #geom_featmap=attention_featmap,
                                                        uv_loc=uv_coord_map,
                                                        sapiens_feature=None)
            
            pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
            pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

            cano_deform_point = pred_point_res + self.query_points 

            pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
            full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

            if iteration < 1000:
                pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
            else:
                pred_scales = pred_scales.permute([0,2,1])

            pred_shs = pred_shs.permute([0,2,1])

            pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
            pred_scales = pred_scales.repeat(1,1,3)

            pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

            #offset_loss = torch.mean(pred_res ** 2)
            #geo_loss = torch.mean(self.geo_feature**2)
            #scale_loss = torch.mean(pred_scales)
            
            if self.deform_on == True:
                full_pred, pred_scales, rotations_batch, offset = self._deformation(full_pred, pred_scales, rotations_batch, pose_batch)
                deform_offset_loss = torch.mean(offset ** 2)

            for batch_index in range(self.batch_size):
                FovX = batch_data['FovX'][batch_index]
                FovY = batch_data['FovY'][batch_index]
                height = batch_data['height'][batch_index]
                width = batch_data['width'][batch_index]
                world_view_transform = batch_data['world_view_transform'][batch_index]
                full_proj_transform = batch_data['full_proj_transform'][batch_index]
                camera_center = batch_data['camera_center'][batch_index]
            

                points = full_pred[batch_index]

                colors = pred_shs[batch_index]
                scales = pred_scales[batch_index] 
            
                
                rendered_image, accumulation = render_batch_custom_background(
                        points=points,
                        shs=None,
                        colors_precomp=colors,
                        rotations=self.fix_rotation,
                        scales=scales,
                        opacity=self.fix_opacity,
                        FovX=FovX,
                        FovY=FovY,
                        height=height,
                        width=width,
                        bg_color=self.background,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,
                        active_sh_degree=0,
                        camera_center=camera_center
                )
                rendered_images.append(rendered_image)
                #accumulations.append(accumulation)

            rendered_images = torch.stack(rendered_images, dim=0).squeeze(0)

            Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(rendered_images, gt_image) / self.batch_size
            ssim_loss = opt.lambda_dssim * (1.0 - ssim(rendered_images, gt_image)) / self.batch_size
            vgg_loss = opt.lambda_lpips * loss_fn_vgg((rendered_images-0.5)*2, (gt_image- 0.5)*2).mean() / self.batch_size
            loss = Ll1 + ssim_loss + vgg_loss

            loss.backward()
            optimizer_smpl.step()
            scheduler.step()

        return pose_b.detach().clone(), pose_r.detach().clone(), trans.detach().clone()



    def eval_stage1(self, batch_data, iteration, pose_data, transl_data):
        
        rendered_images = []
        accumulations = []


        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = pose_data[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=pose_data[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        #sapiens_feature = self.sapiens_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        #attention_featmap = self.attention_net.forward(geom_featmap, sapiens_feature)
        #print("geom_featmap max min: ", geom_featmap.max(),geom_featmap.min())
        #print("uv_coord_map max min: ", uv_coord_map.max(),uv_coord_map.min())
        #print("sapiens_feature max min: ", sapiens_feature.max(),sapiens_feature.min())
        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    #geom_featmap=attention_featmap,
                                                    uv_loc=uv_coord_map,
                                                    sapiens_feature=None)
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
        
        if self.deform_on == True:
            full_pred, pred_scales, rotations_batch, offset = self._deformation(full_pred, pred_scales, rotations_batch, pose_batch)
            deform_offset_loss = torch.mean(offset ** 2)

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index] 
        
            
            rendered_image, accumulation = render_batch_custom_background(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
            )
            rendered_images.append(rendered_image)
            accumulations.append(accumulation)
            

        return torch.stack(rendered_images, dim=0).squeeze(0), torch.stack(accumulations, dim=0).squeeze(0)
##########################################################################################################


    def render_free_stage1(self, batch_data, iteration, total_iteration):
        
        rendered_images = []
        accumulations = []
        pose_data = batch_data['pose_data']
        transl_data = batch_data['transl_data']

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = pose_data[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=pose_data[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        #sapiens_feature = self.sapiens_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        #attention_featmap = self.attention_net.forward(geom_featmap, sapiens_feature)
        #print("geom_featmap max min: ", geom_featmap.max(),geom_featmap.min())
        #print("uv_coord_map max min: ", uv_coord_map.max(),uv_coord_map.min())
        #print("sapiens_feature max min: ", sapiens_feature.max(),sapiens_feature.min())
        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    #geom_featmap=attention_featmap,
                                                    uv_loc=uv_coord_map,
                                                    sapiens_feature=None)
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])
        
        #pred_scales = pred_scales * 1e-1

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        rotations_batch = self.fix_rotation.expand(self.batch_size, -1, -1).contiguous()

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index] 

            #####################rm avatar rectification(deformation model) 2025.04.16##################################
            rotations = rotations_batch[batch_index]
            pose = pose_data[batch_index][3:]
            
            if self.deform_on == True:
                points, scales, rotations, offset = self._deformation(points, scales, rotations, pose, iteration, total_iteration)
                #deform_offset_loss = deform_offset_loss + torch.mean(offset) ** 2
            ############################################################################################################
            
            rendered_image, accumulation = render_batch_custom_background(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=rotations,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
            )
            rendered_images.append(rendered_image)
            accumulations.append(accumulation)
            

        return torch.stack(rendered_images, dim=0).squeeze(0), torch.stack(accumulations, dim=0).squeeze(0)

###########################2025.05.02 rendering##########################
    def render_gaussians(self, batch_data, iteration, total_iteration):
        rendered_images = []
        accumulations = []
        pose_data = batch_data['pose_data']
        transl_data = batch_data['transl_data']

        if self.model_parms.smpl_type == 'smplx':
            leg_angle = 30
            smplx_cpose_param = torch.zeros(1, 165).to('cuda:0')
            smplx_cpose_param[:, 5] =  leg_angle / 180 * math.pi
            smplx_cpose_param[:, 8] = -leg_angle / 180 * math.pi
            #smplx_cpose_param = to_cuda(smplx_cpose_param, device=torch.device('cuda:0'))
            oula_arm_l = transforms.euler_angles_to_matrix(torch.tensor(np.array([[-90, 0, 0]])) / 180 * np.pi, 'XYZ')
            axis_arm_l = transforms.matrix_to_axis_angle(oula_arm_l)
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = smplx_cpose_param[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            leg_angle = 30
            smpl_cpose_param = torch.zeros(1, 72).to('cuda:0')
            smpl_cpose_param[:, 5] =  leg_angle / 180 * math.pi
            smpl_cpose_param[:, 8] = -leg_angle / 180 * math.pi
            #smpl_cpose_param = to_cuda(smpl_cpose_param, device=torch.device('cuda:0'))
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=smpl_cpose_param[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        #sapiens_feature = self.sapiens_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        #attention_featmap = self.attention_net.forward(geom_featmap, sapiens_feature)
        #print("geom_featmap max min: ", geom_featmap.max(),geom_featmap.min())
        #print("uv_coord_map max min: ", uv_coord_map.max(),uv_coord_map.min())
        #print("sapiens_feature max min: ", sapiens_feature.max(),sapiens_feature.min())
        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=None,
                                                    geom_featmap=geom_featmap,
                                                    #geom_featmap=attention_featmap,
                                                    uv_loc=uv_coord_map,
                                                    sapiens_feature=None)
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]
        #full_pred = cano_deform_point

        if iteration < 1000:
            pred_scales = pred_scales.permute([0,2,1]) * 1e-3 * iteration 
        else:
            pred_scales = pred_scales.permute([0,2,1])

        pred_scales = pred_scales * 1e-3

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        rotations_batch = self.fix_rotation.expand(self.batch_size, -1, -1).contiguous()

        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]

            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index] 

            #####################rm avatar rectification(deformation model) 2025.04.16##################################
            rotations = rotations_batch[batch_index]
            pose = pose_data[batch_index][3:]
            
            if self.deform_on == True:
                points, scales, rotations, offset = self._deformation(points, scales, rotations, pose, iteration, total_iteration)
                #deform_offset_loss = deform_offset_loss + torch.mean(offset) ** 2
            ############################################################################################################
            
            rendered_image, accumulation = render_batch_custom_background(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=rotations,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
            )
            rendered_images.append(rendered_image)
            accumulations.append(accumulation)
            

        return torch.stack(rendered_images, dim=0).squeeze(0), torch.stack(accumulations, dim=0).squeeze(0)
###############################################################################################################

    def render_free_stage2(self, batch_data, iteration):
        
        rendered_images = []
        inp_posmap = batch_data['inp_pos_map']
        idx = batch_data['pose_idx']

        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)

        if self.model_parms.smpl_type == 'smplx':
            rest_pose = batch_data['rest_pose']
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = rest_pose[:, :3],
                                                leye_pose=rest_pose[:, 3:6],
                                                reye_pose=rest_pose[:, 6:9],
                                                left_hand_pose= rest_pose[:, 9:54],
                                                right_hand_pose= rest_pose[:, 54:])
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

        pose_featmap = self.pose_encoder(inp_posmap)

        pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                    geom_featmap=geom_featmap,
                                                    uv_loc=uv_coord_map)

        
        
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

        cano_deform_point = pred_point_res + self.query_points

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]


        pred_scales = pred_scales.permute([0,2,1])

        pred_shs = pred_shs.permute([0,2,1])

        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
        # aiap_all_loss = 0
        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        

            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            scales = pred_scales[batch_index]

            rendered_images.append(
                render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=self.fix_rotation,
                    scales=scales,
                    opacity=self.fix_opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            )

        return torch.stack(rendered_images, dim=0)
