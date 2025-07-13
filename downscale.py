import os
import cv2
from pathlib import Path

def resize_images_half(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    count = 0
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to load: {img_path}")
            continue

        h, w = img.shape[:2]
        resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), resized)
        count += 1
        print(f"✅ Resized: {img_path.name} -> {resized.shape[1]}x{resized.shape[0]}")

    print(f"\nDone. {count} images saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save resized images")
    args = parser.parse_args()

    resize_images_half(args.input_dir, args.output_dir)

