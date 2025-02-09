import os
import json
import cv2
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

# Paths (Updated to match primary directories)
DATA_DIR = "../data/"
RAW_VIDEO_PATH = os.path.join(DATA_DIR, "raw_videos/")
PROCESSED_VIDEO_PATH = os.path.join(DATA_DIR, "processed_videos/")
FRAMES_PATH = os.path.join(DATA_DIR, "frames/")
CAPTIONS_PATH = os.path.join(DATA_DIR, "captions/captions.json")
PROCESSED_CAPTIONS_PATH = os.path.join(DATA_DIR, "captions/processed_captions.json")

# Ensure necessary directories exist
os.makedirs(PROCESSED_VIDEO_PATH, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "captions"), exist_ok=True)

# Load captions
if os.path.exists(CAPTIONS_PATH):
    with open(CAPTIONS_PATH, "r") as f:
        captions = json.load(f)
else:
    print(f"Warning: Captions file not found at {CAPTIONS_PATH}. Captions will be missing.")
    captions = {}

# Dictionary to store captions for processed videos
processed_captions = {}


def split_videos(video_path, output_path, video_caption):
    """Splits a video into scenes using the latest PySceneDetect API and saves captions."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))  # Tune threshold if needed

    # Detect scenes
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    print(f"Detected {len(scenes)} scenes in {video_path}")

    # Split scenes and save them
    video_basename = os.path.basename(video_path).replace('.mp4', '')  # Remove extension
    split_video_ffmpeg(video_path, scenes, output_dir=output_path)

    # Manually construct filenames since split_video_ffmpeg does not return them
    for i, _ in enumerate(scenes):
        scene_filename = f"{video_basename}_scene_{i + 1}.mp4"
        processed_captions[scene_filename] = video_caption  # Associate caption


def extract_keyframes(video_path, output_path):
    """Extracts keyframes from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Processing {video_path}: {frame_count} frames at {fps} FPS.")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 30th frame as a keyframe
        if count % 30 == 0:
            frame_filename = f"{os.path.basename(video_path).replace('.mp4', '')}_frame_{count}.jpg"
            frame_output_path = os.path.join(output_path, frame_filename)
            cv2.imwrite(frame_output_path, frame)
            print(f"Saved keyframe: {frame_output_path}")

        count += 1

    cap.release()


def main():
    for video_file in os.listdir(RAW_VIDEO_PATH):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(RAW_VIDEO_PATH, video_file)

            # Retrieve caption for the video
            video_caption = captions.get(video_file, "No caption available")

            print(f"Processing {video_file} with caption: {video_caption}")

            # Process video
            split_videos(video_path, PROCESSED_VIDEO_PATH, video_caption)
            extract_keyframes(video_path, FRAMES_PATH)

    # Save processed captions
    with open(PROCESSED_CAPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_captions, f, indent=4)
    print(f"Saved processed captions to {PROCESSED_CAPTIONS_PATH}")


if __name__ == "__main__":
    main()
