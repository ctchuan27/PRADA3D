<div align="center">

# <b>PRADA3D</b>: Photorealistic and Real-Time Animatable 3D Gaussian Avatar Reconstruction with Deformation Distillation and Dual Rectification

</div>

## 環境建置
首先進入想要的資料夾，例如：
```bash
cd Tun-Chuan
```

下載 PRADA3D 的 GitHub 並進入資料夾：
```bash
git clone https://github.com/ctchuan27/PRADA3D.git
cd PRADA3D
```

建立環境並安裝套件：
```bash
conda env create --file environment.yml
conda activate PRADA3D_env
```

編譯 3DGS 必要模組：`diff-gaussian-rasterization` 與 `simple-knn`：
```bash
cd submodules
git clone https://github.com/jkulhanek/fork-diff-gaussian-rasterization.git
mv fork-diff-gaussian-rasterization diff_gaussian_rasterization
cd diff_gaussian_rasterization
python setup.py install
cd ..
cd simple-knn
python setup.py install
cd ..
cd ..
pip install numpy==1.23.1
```

⚠️ 注意：這裡使用的是可分割背景的 fork 版本 `diff-gaussian-rasterization`，不同於原版 3DGS。更多說明請參考：  
https://github.com/graphdeco-inria/gaussian-splatting/issues/542

---

## 下載模型與資料 
- **SMPL/SMPL-X 模型**：請先註冊並下載 [SMPL](https://smpl.is.tue.mpg.de/) 與 [SMPL-X](https://smpl-x.is.tue.mpg.de/)，並放置於 `assets/smpl_files`。  

- **資料集與模型檔**：請從 [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/lx_hu_hit_edu_cn/EsGcL5JGKhVGnaAtJ-rb1sQBR4MwkdJ9EWqJBIdd2mpi2w?e=KnloBM) 下載 `assets.zip`、`gs_data.zip` 與 `pretrained_models.zip`。  
  - `assets.zip` → 解壓縮至專案內對應資料夾  
  - `gs_data.zip` → 解壓縮至 `dataset`  
  - `pretrained_models.zip` → 解壓縮至 `output`  

資料夾結構如下：
```
PRADA3D
 └── assets
     └── smpl_files
           └── smpl
               ├── SMPL_FEMALE.pkl
               ├── SMPL_MALE.pkl
               └── SMPL_NEUTRAL.pkl
           └── smplx
               ├── SMPLX_FEMALE.npz
               ├── SMPLX_MALE.npz
               └── SMPLX_NEUTRAL.npz
 └── dataset
 └── output
```

---

## 指令

通用模板（自行修改 -s 與 -m）

### 訓練指令
```bash
python train.py \
    -s <path_to_dataset> \
    -m <path_to_training_result> \
    --train_stage 1 \
    --epoch 300 \
    --smpl_type smpl \
    --pose_op_start_iter 10 \
    --lpips_start_iter 30 \
    --position_lr_init 0.0008
```
使用 LHM（NPDD）請加： `--lhm`

### 渲染指令

```bash
python render_novel_pose.py \
    -s <path_to_dataset> \
    -m <path_to_training_result> \
    --train_stage 1 \
    --epoch 300 \
    --smpl_type smpl
```

### evaluation指令
```bash
python eval.py \
    -s <path_to_dataset> \
    -m <path_to_training_result> \
    --train_stage 1 \
    --epoch 300 \
    --smpl_type smpl
```

### 即時動畫指令(live demo)
```bash
python realtime_live_demo.py \
    -s <path_to_dataset> \
    -m <path_to_training_result> \
    --train_stage 1 \
    --epoch 300 \
    --smpl_type smpl \
    --webcam_id 3
```
根據連接的webcam代碼改變 `--webcam_id`

## 訓練指令範例
### People Snapshot dataset (with LHM / NPDD)

Subject: m3c
```bash
python train.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
python render_novel_pose.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v4_0.0008 --train_stage 1 --epoch 300 --lhm
python eval.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v4_0.0008 --train_stage 1 --epoch 300 --lhm --webcam_id 3
```
Subject: m4c
```bash
python train.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
python render_novel_pose.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v4_0.0008 --train_stage 1 --epoch 300 --lhm
python eval.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v4_0.0008 --train_stage 1 --epoch 300 --lhm --webcam_id 3
```
Subject: f3c
```bash
python train.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
python render_novel_pose.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v4_0.0008 --train_stage 1 --epoch 300 --lhm
python eval.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v4_0.0008 --train_stage 1 --epoch 300 --lhm --webcam_id 3
```
Subject: f4c
```bash
python train.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
python render_novel_pose.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v4_0.0008 --train_stage 1 --epoch 300 --lhm
python eval.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v4_0.0008 --train_stage 1 --epoch 300 --lhm --webcam_id 3
```

### Custom dataset

Subject: 傅老師
```bash
python train.py -s ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 墩權（帽踢）
```bash
python train.py -s ./custom_dataset/tunchuan/hoodie_v2 -m output/custom/tunchuan/hoodie_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/tunchuan/hoodie_v2 -m output/custom/tunchuan/hoodie_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/tunchuan/hoodie_v2 -m output/custom/tunchuan/hoodie_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/tunchuan/hoodie_v2 -m output/custom/tunchuan/hoodie_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 墩權（條紋長袖）
```bash
python train.py -s ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 鳳儀
```bash
python train.py -s ./custom_dataset/fengyi/fengyi_alldown/origres -m output/custom/fengyi/fengyi_alldown/origres/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/fengyi/fengyi_alldown/origres -m output/custom/fengyi/fengyi_alldown/origres/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/fengyi/fengyi_alldown/origres -m output/custom/fengyi/fengyi_alldown/origres/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/fengyi/fengyi_alldown/origres -m output/custom/fengyi/fengyi_alldown/origres/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 靜睿
```bash
python train.py -s ./custom_dataset/jingrei -m output/custom/jingrei/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/jingrei -m output/custom/jingrei/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/jingrei -m output/custom/jingrei/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/jingrei -m output/custom/jingrei/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 詩婷
```bash
python train.py -s ./custom_dataset/shitin -m output/custom/shitin/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/shitin -m output/custom/shitin/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/shitin -m output/custom/shitin/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/shitin -m output/custom/shitin/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 婷安
```bash
python train.py -s ./custom_dataset/tingan -m output/custom/tingan/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/tingan -m output/custom/tingan/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/tingan -m output/custom/tingan/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/tingan -m output/custom/tingan/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 資工同學（student_1）
```bash
python train.py -s ./custom_dataset/student_1 -m output/custom/student_1/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/student_1 -m output/custom/student_1/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/student_1 -m output/custom/student_1/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/student_1 -m output/custom/student_1/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```
Subject: 資工同學（student_4）
```bash
python train.py -s ./custom_dataset/student_4 -m output/custom/student_4/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
python render_novel_pose.py -s ./custom_dataset/student_4 -m output/custom/student_4/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python eval.py -s ./custom_dataset/student_4 -m output/custom/student_4/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl
python realtime_live_demo.py --source_path ./custom_dataset/student_4 -m output/custom/student_4/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --webcam_id 3
```


## 使用自訂影片資料

### 前處理(未完成待修改)
首先將影片分割:
```bash
python video_to_frame.py \
  --video_file <影片位置> \
  --output_dir <output image folder> \
  --downsample <frame取樣> default=2, help="Frame downsample rate (keep 1 every N frames)" \
  --downscale <image大小調整> default=1.0, help="Resolution downscale factor (e.g. 0.5 for half size)" \
  --rotation <image旋轉> choices=[0, 90, 180, 270], default=0, help="Rotation angle: 0=none, 90=clockwise, 180=flip, 270=counter-clockwise"
```
舉例:
```bash
python video_to_frame.py \
  --video_file ./custom_dataset/tunchuan/stripe/sam.MOV \
  --output_dir ./custom_dataset/tunchuan/stripe/images \
  --downsample 2 \
  --downscale 0.5 \
  --rotation 90
```
使用ReFit預測3D pose，如壞掉要安裝參考:https://github.com/yufu-wang/ReFit \
將照片複製到ReFit資料夾 \
舉例:
```bash
cd ..
mkdir -p ReFit/custom_dataset/tunchuan/stripe/images
cp -r PRADA3D/custom_dataset/tunchuan/stripe/images ReFit/custom_dataset/tunchuan/stripe
```
接著預測3D pose，舉例:
```bash
conda deactivate
conda activate refit
cd ReFit
python scripts/process_folder.py --imagedir custom_dataset/tunchuan/stripe --viz_results
```

資料夾結構應如下：
```
ReFit/custom_dataset/tunchuan/stripe
 ├── images
 ├── cameras.npz
 └── poses_optimized.npz
```

複製回PRADA3D資料夾:
```bash
cd ..
cp -r ReFit/custom_dataset/tunchuan/stripe PRADA3D/custom_dataset/tunchuan
conda deactivate
```

用langsam分割mask，如壞掉安裝參考: https://github.com/luca-medeiros/lang-segment-anything
```bash
cd PRADA3D
conda activate langsam
python mask_seg.py \
  --image_dir ./custom_dataset/tunchuan/stripe/images \
  --mask_dir ./custom_dataset/tunchuan/stripe/masks \
```

接著資料夾結構應如下：
```
PRADA3D/custom_dataset/tunchuan/stripe
 ├── images
 ├── masks
 ├── cameras.npz
 └── poses_optimized.npz
```

接著轉換到訓練需要格式:
```bash
conda deactivate
conda activate PRADA3D
python scripts/data_preparation.py \
  --data_path ./custom_dataset/tunchuan/stripe \
```

接著資料夾結構應如下：
```
PRADA3D/custom_dataset/tunchuan/stripe
 ├── images
 ├── masks
 ├── train
     ├── images
     ├── masks
     ├── cam_parms.npz
     ├── cano_smpl.obj
     ├── query_posemap_512_cano_smpl.npz
     ├── smpl_cano_joint_mat.pth
     └── smpl_parms.pth
 ├── test
     ├── images
     ├── masks
     ├── cam_parms.npz
     ├── cano_smpl.obj
     ├── query_posemap_512_cano_smpl.npz
     ├── smpl_cano_joint_mat.pth
     └── smpl_parms.pth
 ├── cameras.npz
 └── poses_optimized.npz
```




使用 [InstantAvatar](https://github.com/tijiang13/InstantAvatar) 提供的腳本產生遮罩與姿勢檔：
```bash

scripts/custom/process-sequence.sh
```



轉換 ROMP 姿勢格式（需修改第 50、51 行路徑）：
```bash
cd scripts & python sample_romp2gsavatar.py
```

生成標準姿勢的 position map（需修改對應路徑）：
```bash
python gen_pose_map_cano_smpl.py
```


