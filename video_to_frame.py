import cv2
import os
import argparse

# Function to extract frames from a video with arguments
def extract_frames(video_file, output_directory, downsample, downscale, rotation):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    # Create an output folder if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation if specified
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Downsample: only keep every N-th frame
        if frame_count % downsample == 0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * downscale), int(w * downscale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            output_file = os.path.join(output_directory, f"{int(frame_count / downsample):08d}.png")
            cv2.imwrite(output_file, frame)
            print(f"Frame {int(frame_count / downsample)} saved as {output_file}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video_file", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--downsample", type=int, default=2, help="Frame downsample rate (keep 1 every N frames)")
    parser.add_argument("--downscale", type=float, default=1.0, help="Resolution downscale factor (e.g. 0.5 for half size)")
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotation angle: 0=none, 90=clockwise, 180=flip, 270=counter-clockwise")

    args = parser.parse_args()

    extract_frames(args.video_file, args.output_dir, args.downsample, args.downscale, args.rotation)

