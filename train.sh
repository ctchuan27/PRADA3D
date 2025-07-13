#!/bin/bash

#python train.py -s ./tunchuan_dataset/v4_refit/sam -m output/sam_v4_refit_stage1 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30

#python train.py -s ./default_dataset/m4c_processed -m output/m4c_processed_v2 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./default_dataset/dynvideo_female -m output/dynvideo_female --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./default_dataset/dynvideo_male -m output/dynvideo_male --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 30


#python train.py -s ./default_dataset/m4c_refit -m output/m4c_refit_v2 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20
#python train.py -s ./tunchuan_dataset/newvideo_v1 -m output/new_video_v2_stage1 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20
#python train.py -s ./tunchuan_dataset/v4_refit/sam -m output/sam_v4_refit_stage1_v4 --train_stage 1 --pose_op_start_iter 10 --lpips_start_iter 20

#python train.py -s ./default_dataset/m4c_processed -m output/m4c_processed_v2 --train_stage 1
#python train.py -s ./default_dataset/m3c_processed -m output/m3c_processed_v2 --train_stage 1
#python train.py -s ./default_dataset/f4c_processed -m output/f4c_processed_v2 --train_stage 1
#python train.py -s ./default_dataset/f3c_processed -m output/f3c_processed_v2 --train_stage 1



#python train.py -s custom_dataset/neuman/refit/bike -m output/neuman/bike_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s custom_dataset/neuman/refit/citron -m output/neuman/citron_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s custom_dataset/neuman/refit/jogging -m output/neuman/jogging_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s custom_dataset/neuman/refit/seattle -m output/neuman/seattle_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 3#0

#python scripts/export_optimized_testpose.py --output_path custom_dataset/neuman/refit/bike/test/smpl_parms.pth --net_path output/neuman/bike_refit/v2/net/iteration_300 --smpl_parms_path custom_dataset/neuman/refit/bike/train
#python scripts/export_optimized_testpose.py --output_path custom_dataset/neuman/refit/citron/test/smpl_parms.pth --net_path output/neuman/citron_refit/v2/net/iteration_300 --smpl_parms_path custom_dataset/neuman/refit/citron/train
#python scripts/export_optimized_testpose.py --output_path custom_dataset/neuman/refit/jogging/test/smpl_parms.pth --net_path output/neuman/jogging_refit/v2/net/iteration_300 --smpl_parms_path custom_dataset/neuman/refit/jogging/train
#python scripts/export_optimized_testpose.py --output_path custom_dataset/neuman/refit/seattle/test/smpl_parms.pth --net_path output/neuman/seattle_refit/v2/net/iteration_300 --smpl_parms_path custom_dataset/neuman/refit/seattle/train

#python eval.py -s custom_dataset/neuman/refit/bike -m output/neuman/bike_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl
#python eval.py -s custom_dataset/neuman/refit/citron -m output/neuman/citron_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl
#python eval.py -s custom_dataset/neuman/refit/jogging -m output/neuman/jogging_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl
#python eval.py -s custom_dataset/neuman/refit/seattle -m output/neuman/seattle_refit/v2 --train_stage 1 --epoch 300 --smpl_type smpl


#python train.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v3_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
#python train.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v3_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
#python train.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v3_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
#python train.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v3_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008


#python train.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v2 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm
#python train.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v2 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm
#python eval.py -s lhm_dataset/people_snapshot/m3c_processed -m output/lhm/people_snapshot/m3c_v1 --train_stage 1 --epoch 300 --smpl_type smpl --lhm
#python eval.py -s lhm_dataset/people_snapshot/m4c_processed -m output/lhm/people_snapshot/m4c_v1 --train_stage 1 --epoch 300 --smpl_type smpl --lhm
#python scripts/export_optimized_testpose.py --output_path lhm_dataset/people_snapshot/m3c_processed/test/smpl_parms.pth --net_path output/lhm/people_snapshot/m3c_v1/net/iteration_300 --smpl_parms_path lhm_dataset/people_snapshot/m3c_processed/train

#python train.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v2 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm
#python train.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v2 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm


#python train.py -s default_dataset/people_snapshot/m4c_processed -m output/default/people_snapshot/m4c_v1 --train_stage 1 --epoch 300 --lpips_start_iter 30
#python train.py -s default_dataset/people_snapshot/f3c_processed -m output/default/people_snapshot/f3c_v1 --train_stage 1 --epoch 300 --lpips_start_iter 30
#python train.py -s default_dataset/people_snapshot/f4c_processed -m output/default/people_snapshot/f4c_v1 --train_stage 1 --epoch 300 --lpips_start_iter 30

#python train.py -s sr_dataset/people_snapshot/m3c -m output/sr/people_snapshot/m3c_v1 --train_stage 1 --epoch 300 --smpl_type smplx --lpips_start_iter 30 --pose_op_start_iter 10 --sr

#python train.py -s ./custom_dataset/tunchuan/hoodie -m output/custom/tunchuan/hoodie_v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./custom_dataset/tunchuan/stripe_refit -m output/custom/tunchuan/stripe_v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30

#python train.py -s ./custom_dataset/tunchuan/hoodie -m output/custom/tunchuan/hoodie_v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008

#python train.py -s ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_v3_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008

#python train.py -s sr_dataset/people_snapshot/m3c -m output/sr/people_snapshot/m3c_v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smplx --lpips_start_iter 30 --pose_op_start_iter 10 --sr --position_lr_init 0.0008

#python train.py -s ./custom_dataset/tunchuan/stripe -m output/custom/tunchuan/stripe_v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008

#python render_novel_pose.py -s lhm_dataset/people_snapshot/m3c_processed -m output/lhm/people_snapshot/m3c_v1 --train_stage 1 --epoch 300 --smpl_type smpl --lhm --test_folder lhm_dataset/people_snapshot/m4c_processed/train

#python train.py -s default_dataset/people_snapshot/m4c_processed -m output/awful_exp/people_snapshot/m4c_v1 --train_stage 1 --epoch 200 --query_posmap_size 256 --inp_posmap_size 256
#python train.py -s default_dataset/people_snapshot/f3c_processed -m output/awful_exp/people_snapshot/f3c_v1 --train_stage 1 --epoch 200 --query_posmap_size 256 --inp_posmap_size 256
#python train.py -s default_dataset/people_snapshot/f4c_processed -m output/awful_exp/people_snapshot/f4c_v1 --train_stage 1 --epoch 200 --query_posmap_size 256 --inp_posmap_size 256

#python eval.py -s default_dataset/people_snapshot/m4c_processed -m output/awful_exp/people_snapshot/m4c_v1 --train_stage 1 --epoch 200
#python eval.py -s default_dataset/people_snapshot/f3c_processed -m output/awful_exp/people_snapshot/f3c_v1 --train_stage 1 --epoch 200
#python eval.py -s default_dataset/people_snapshot/f4c_processed -m output/awful_exp/people_snapshot/f4c_v1 --train_stage 1 --epoch 200

#python eval.py -s default_dataset/people_snapshot/m3c_processed -m output/default/people_snapshot/m3c_v1 --train_stage 1 --epoch 300
#python eval.py -s default_dataset/people_snapshot/m4c_processed -m output/default/people_snapshot/m4c_v1 --train_stage 1 --epoch 300
#python eval.py -s default_dataset/people_snapshot/f3c_processed -m output/default/people_snapshot/f3c_v1 --train_stage 1 --epoch 300
#python eval.py -s default_dataset/people_snapshot/f4c_processed -m output/default/people_snapshot/f4c_v1 --train_stage 1 --epoch 300

#python train.py -s ./custom_dataset/fengyi -m output/custom/fengyi/v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./custom_dataset/fengyi -m output/custom/fengyi/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
#python train.py -s ./custom_dataset/fengyi_lowres -m output/custom/fengyi_lowres/v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
#python train.py -s ./custom_dataset/fengyi_lowres -m output/custom/fengyi_lowres/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008

python train.py -s ./custom_dataset/fu -m output/custom/fu/v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
python train.py -s ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python train.py -s ./custom_dataset/fu_lowres -m output/custom/fu_lowres/v1 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30
python train.py -s ./custom_dataset/fu_lowres -m output/custom/fu_lowres/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008



python train.py -s ./custom_dataset/tunchuan/stripe -m output/custom/tunchuan/stripe_v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008


