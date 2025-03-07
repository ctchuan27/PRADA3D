from pathlib import Path
import tqdm
import cv2

def set_camera(filename: str, width: int, height: int, 
               fps: float = 20.0, fourcc: str = 'mp4v') -> cv2.VideoWriter:
    return cv2.VideoWriter(
        filename,
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=(width, height)
    )

render_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/tunchuan_dataset/new_video_v1/train/images'
batch_images = sorted(Path(render_path).glob('*.png'))
video_output_path = '/home/enjhih/Tun-Chuan/GaussianAvatar/tunchuan_dataset/new_video_v1/train/images/video.mp4'

camera = None
for ip in tqdm.tqdm(batch_images):
    frame = cv2.imread(ip.as_posix())
    
    # process ...
    
    if camera is None:
        h, w, _ = frame.shape
        camera = set_camera(video_output_path, w, h)
    camera.write(frame)
camera.release()
