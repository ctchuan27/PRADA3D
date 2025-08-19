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
conda activate PRADA3D
```

編譯 3DGS 必要模組：`diff-gaussian-rasterization` 與 `simple-knn`：
```bash
git clone https://github.com/jkulhanek/fork-diff-gaussian-rasterization.git
mv fork-diff-gaussian-rasterization diff-gaussian-rasterization
pip install diff-gaussian-rasterization
pip install simple-knn
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

## 在 People Snapshot 資料集上執行
以 `m4c_processed` 為例。  

訓練：
```bash
python train.py -s $gs_data_path/m4c_processed -m output/m4c_processed --train_stage 1
```

評估：
```bash
python eval.py -s $gs_data_path/m4c_processed -m output/m4c_processed --epoch 200
```

渲染新姿勢：
```bash
python render_novel_pose.py -s $gs_data_path/m4c_processed -m output/m4c_processed --epoch 200
```

---

## 使用自訂影片資料

### 前處理
使用 [InstantAvatar](https://github.com/tijiang13/InstantAvatar) 提供的腳本產生遮罩與姿勢檔：
```bash
scripts/custom/process-sequence.sh
```

資料夾結構應如下：
```
custom_data
 ├── images
 ├── masks
 ├── cameras.npz
 └── poses_optimized.npz
```

轉換 ROMP 姿勢格式（需修改第 50、51 行路徑）：
```bash
cd scripts & python sample_romp2gsavatar.py
```

生成標準姿勢的 position map（需修改對應路徑）：
```bash
python gen_pose_map_cano_smpl.py
```

---

## 通用訓練指令模板與範例

通用模板（自行修改 -s 與 -m）

```bash
python train.py \
    -s <path_to_dataset>/<subject_folder> \
    -m output/<category>/<subject_name>/<exp_name> \
    --train_stage 1 \
    --epoch 300 \
    --smpl_type smpl \
    --pose_op_start_iter 10 \
    --lpips_start_iter 30 \
    --position_lr_init 0.0008
```

使用 LHM（NPDD）請加： --lhm


People Snapshot dataset（with LHM / NPDD）

```bash
# Subject: m3c
python train.py -s lhm_dataset/people_snapshot_v2/m3c_processed -m output/lhm/people_snapshot/m3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008

# Subject: m4c
python train.py -s lhm_dataset/people_snapshot_v2/m4c_processed -m output/lhm/people_snapshot/m4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008

# Subject: f3c
python train.py -s lhm_dataset/people_snapshot_v2/f3c_processed -m output/lhm/people_snapshot/f3c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008

# Subject: f4c
python train.py -s lhm_dataset/people_snapshot_v2/f4c_processed -m output/lhm/people_snapshot/f4c_v4_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --lpips_start_iter 30 --pose_op_start_iter 10 --lhm --position_lr_init 0.0008
```

Custom dataset:

Subject: 傅老師
```bash
python train.py -s ./custom_dataset/fu -m output/custom/fu/v2_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 墩權（帽踢）
```bash
python train.py -s ./custom_dataset/tunchuan/hoodie_v2 -m output/custom/tunchuan/hoodie_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 墩權（條紋長袖）
```bash
python train.py -s ./custom_dataset/tunchuan/stripe_v2 -m output/custom/tunchuan/stripe_cut/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 鳳儀
```bash
python train.py -s ./custom_dataset/fengyi/fengyi_alldown/origres -m output/custom/fengyi/fengyi_alldown/origres/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 靜睿
```bash
python train.py -s ./custom_dataset/jingrei -m output/custom/jingrei/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 詩婷
```bash
python train.py -s ./custom_dataset/shitin -m output/custom/shitin/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 婷安
```bash
python train.py -s ./custom_dataset/tingan -m output/custom/tingan/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 資工同學（student_1）
```bash
python train.py -s ./custom_dataset/student_1 -m output/custom/student_1/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
Subject: 資工同學（student_4）
```bash
python train.py -s ./custom_dataset/student_4 -m output/custom/student_4/v1_0.0008 --train_stage 1 --epoch 300 --smpl_type smpl --pose_op_start_iter 10 --lpips_start_iter 30 --position_lr_init 0.0008
```
