import os
import shutil

def downsample_images_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, step=4):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for i, filename in enumerate(image_filenames):
        if i % step == 0:
            src_img = os.path.join(image_dir, filename)
            src_mask = os.path.join(mask_dir, filename)
            dst_img = os.path.join(output_image_dir, filename)
            dst_mask = os.path.join(output_mask_dir, filename)

            shutil.copy(src_img, dst_img)
            shutil.copy(src_mask, dst_mask)

    print(f"Downsampled {len(image_filenames)//step} image-mask pairs saved.")

# 修改成你的路徑
image_dir = "/home/enjhih/Tun-Chuan/GaussianAvatar/custom_dataset/tunchuan/hoodie/images"
mask_dir = "/home/enjhih/Tun-Chuan/GaussianAvatar/custom_dataset/tunchuan/hoodie/masks"
output_image_dir = "/home/enjhih/Tun-Chuan/GaussianAvatar/custom_dataset/tunchuan/hoodie_v2/images"
output_mask_dir = "/home/enjhih/Tun-Chuan/GaussianAvatar/custom_dataset/tunchuan/hoodie_v2/masks"

downsample_images_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, step=4)
