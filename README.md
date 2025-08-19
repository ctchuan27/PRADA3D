<div align="center">

# <b>PRADA3D</b>: Photorealistic and Real-Time Animatable 3D Gaussian Avatar Reconstruction with Deformation Distillation and Dual Rectification

</div>

## 環境建置
建立環境並安裝套件：
```bash
conda env create --file environment.yml
conda activate gs-avatar
```
接著依照 [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 的指引，編譯以下模組： `diff-gaussian-rasterization` 與 `simple-knn`。

## 下載模型與資料 
SMPL/SMPL-X 模型：請先註冊並下載 [SMPL](https://smpl.is.tue.mpg.de/) 與 [SMPL-X](https://smpl-x.is.tue.mpg.de/)，並放置於 `assets/smpl_files`。資料夾結構如下：
```
smpl_files
 └── smpl
   ├── SMPL_FEMALE.pkl
   ├── SMPL_MALE.pkl
   └── SMPL_NEUTRAL.pkl
 └── smplx
   ├── SMPLX_FEMALE.npz
   ├── SMPLX_MALE.npz
   └── SMPLX_NEUTRAL.npz
```

資料集與模型檔：請從 [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/lx_hu_hit_edu_cn/EsGcL5JGKhVGnaAtJ-rb1sQBR4MwkdJ9EWqJBIdd2mpi2w?e=KnloBM) 下載 `assets.zip`、`gs_data.zip` 與 `pretrained_models.zip`。將 `assets.zip` 解壓縮至專案對應資料夾，`gs_data.zip` 解壓縮至 `$gs_data_path`，`pretrained_models.zip` 解壓縮至 `$pretrained_models_path`。

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

## 使用自訂影片資料

### 前處理
遮罩與姿勢檔：使用 [InstantAvatar](https://github.com/tijiang13/InstantAvatar) 提供的腳本：
```bash
scripts/custom/process-sequence.sh
```
資料夾結構應如下：
```
smpl_files
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

### Stage 1 訓練
```bash
cd .. & python train.py -s $path_to_data/$subject -m output/{$subject}_stage1 --train_stage 1 --pose_op_start_iter 10
```

### Stage 2 訓練
匯出預測 SMPL：
```bash
cd scripts & python export_stage_1_smpl.py
```

視覺化優化後的 SMPL（可選）：
```bash
python render_pred_smpl.py
```

生成預測的 position map：
```bash
python gen_pose_map_our_smpl.py
```

開始 Stage 2 訓練：
```bash
cd .. & python train.py -s $path_to_data/$subject -m output/{$subject}_stage2 --train_stage 2 --stage1_out_path $path_to_stage1_net_save_path
```

## 致謝
本專案建立於以下開源程式碼基礎上：  
- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)  
- [POP](https://github.com/qianlim/POP)  
- [HumanNeRF](https://github.com/chungyiweng/humannerf)  
- [InstantAvatar](https://github.com/tijiang13/InstantAvatar)  
