import cv2
import os

# Function to extract frames from a video until reaching the desired frame count
def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    frame_rate = 30  # Desired frame rate (1 frame every 0.5 seconds)
    frame_count = 0
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"./custom_dataset/tunchuan/stripe_v3/images"
    os.makedirs(output_directory, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #output_file = f"{output_directory}/{int(frame_count):08d}.png"
        output_file = f"{output_directory}/{int(frame_count/2):08d}.png"
        downscale=1
        if frame_count % 2 == 0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * downscale), int(w * downscale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_file, frame)
            print(f"Frame {int(frame_count/2)} has been extracted and saved as {output_file}")
        #cv2.imwrite(output_file, frame)
        #print(f"Frame {frame_count} has been extracted and saved as {output_file}")
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "./custom_dataset/tunchuan/stripe/sam.MOV"  # Replace with your video's name
    
    extract_frames(video_file)
