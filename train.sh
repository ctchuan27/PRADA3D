#!/bin/bash

#python train.py -s ./tunchuan_dataset/v4_refit/sam -m output/sam_v4_refit_stage1 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30

#python train.py -s ./default_dataset/m4c_processed -m output/m4c_processed_v2 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./default_dataset/dynvideo_female -m output/dynvideo_female --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./default_dataset/dynvideo_male -m output/dynvideo_male --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30


#python train.py -s ./default_dataset/m4c_refit -m output/m4c_refit_v2 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20
#python train.py -s ./tunchuan_dataset/newvideo_v1 -m output/new_video_v2_stage1 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20
#python train.py -s ./tunchuan_dataset/v4_refit/sam -m output/sam_v4_refit_stage1_v4 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20

python train.py -s ./default_dataset/m4c_processed -m output/m4c_processed_v2 --train_stage 1
python train.py -s ./default_dataset/m3c_processed -m output/m3c_processed_v2 --train_stage 1
python train.py -s ./default_dataset/f4c_processed -m output/f4c_processed_v2 --train_stage 1
python train.py -s ./default_dataset/f3c_processed -m output/f3c_processed_v2 --train_stage 1
