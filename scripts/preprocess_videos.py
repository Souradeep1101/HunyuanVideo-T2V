import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Paths
RAW_VIDEO_PATH = "../data/raw_videos/"
PROCESSED_VIDEO_PATH = "../data/processed_videos/"
FRAMES_PATH = "../data/frames/"

# Ensure output directories exist
os.makedirs(PROCESSED_VIDEO_PATH, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)


def split_videos(video_path, output_path):
    """Splits a video into scenes using PySceneDetect."""
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))  # Tune threshold if needed

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get detected scenes
    scenes = scene_manager.get_scene_list()
    print(f"Detected {len(scenes)} scenes in {video_path}")

    # Split and save scenes
    for i, scene in enumerate(scenes):
        start_frame, end_frame = scene
        scene_output_path = os.path.join(output_path, f"scene_{i + 1}.mp4")
        video_manager.save_video(scene_output_path, start_frame, end_frame)
        print(f"Saved scene: {scene_output_path}")


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
            frame_output_path = os.path.join(output_path, f"frame_{count}.jpg")
            cv2.imwrite(frame_output_path, frame)
            print(f"Saved keyframe: {frame_output_path}")
        count += 1

    cap.release()


def main():
    # Iterate through raw videos
    for video_file in os.listdir(RAW_VIDEO_PATH):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(RAW_VIDEO_PATH, video_file)

            # Process video
            split_videos(video_path, PROCESSED_VIDEO_PATH)
            extract_keyframes(video_path, FRAMES_PATH)


if __name__ == "__main__":
    main()
