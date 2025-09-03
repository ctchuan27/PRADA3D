import os
import cv2
import torch
import argparse
from tqdm import tqdm
from lang_sam import LangSAM
from PIL import Image
import numpy as np


def run_langsam(sr_dir, mask_dir, sam_ckpt, prompt):
    os.makedirs(mask_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LangSAM()

    image_list = sorted([f for f in os.listdir(sr_dir) if f.lower().endswith(('.png', '.jpg'))])
    for fname in tqdm(image_list, desc="LangSAM Segmentation"):
        img_path = os.path.join(sr_dir, fname)
        image = cv2.imread(img_path)

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        results = model.predict([image_pil], [prompt])
        mask = results[0]
        if isinstance(mask, dict):
            mask = mask["masks"]

        mask_0 = np.squeeze(mask[0])
        if mask.shape[0] > 1:
            mask_1 = np.squeeze(mask[1])
            mask = mask_0 + mask_1
        else:
            mask = mask_0

        mask = (mask > 0).astype(np.uint8) * 255  # binary mask

        # Save binary mask
        cv2.imwrite(os.path.join(mask_dir, fname), mask)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory of input images")
    parser.add_argument("--mask_dir", required=True, help="Directory to save binary masks")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--prompt", default="person.", help="Text prompt for LangSAM")
    args = parser.parse_args()

    run_langsam(args.image_dir, args.mask_dir, args.sam_checkpoint, args.prompt)


if __name__ == "__main__":
    main()

