import cv2
import os
from pathlib import Path

# --- CONFIGURATION ---
# Path to the raw videos you downloaded
REAL_VIDEO_PATH = "data/FaceForensics++/original_sequences/youtube/c23/videos"
FAKE_VIDEO_PATH = "data/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos"

# Path where images will be saved
OUTPUT_REAL_PATH = "data/real"
OUTPUT_FAKE_PATH = "data/fake"

def extract_frames(video_path, output_dir, max_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_count // max_frames)
    
    frame_num = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_num % interval == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1
            
        frame_num += 1
    cap.release()

def process_dataset(input_dir, output_dir):
    print(f"Processing videos from: {input_dir}")
    video_files = list(Path(input_dir).rglob("*.mp4"))
    
    for i, video in enumerate(video_files):
        # Save frames in a subfolder named after the video
        video_name = video.stem
        save_path = os.path.join(output_dir, video_name)
        extract_frames(str(video), save_path)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(video_files)} videos")

if __name__ == "__main__":
    process_dataset(REAL_VIDEO_PATH, OUTPUT_REAL_PATH)
    process_dataset(FAKE_VIDEO_PATH, OUTPUT_FAKE_PATH)
    print("Preprocessing Complete.")