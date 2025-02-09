import os
import requests
import json
from tqdm import tqdm
from datasets import load_dataset

# Paths
DATA_DIR = "../data/"
RAW_VIDEO_PATH = os.path.join(DATA_DIR, "raw_videos/")
CAPTIONS_PATH = os.path.join(DATA_DIR, "captions/captions.json")

# Ensure necessary directories exist
os.makedirs(RAW_VIDEO_PATH, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "captions"), exist_ok=True)


def download_video(video_url, output_path):
    """Download a video from the given URL."""
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download {video_url}: {e}")


def download_filtered_webvid(dataset, output_dir, captions_path, min_resolution=720):
    """Download high-quality videos and save their captions."""
    captions = {}

    for sample in tqdm(dataset, desc="Downloading high-quality WebVid videos"):
        if "contentUrl" not in sample or "name" not in sample:
            print(f"Skipping sample with missing keys: {sample}")
            continue

        video_url = sample["contentUrl"]  # ✅ Correct key for video URL
        caption = sample["name"]  # ✅ Correct key for caption

        # Remove resolution filtering if unnecessary
        # Or use this as a fallback for missing resolution
        width, height = sample.get("width", 1280), sample.get("height", 720)
        if min(width, height) < min_resolution:
            print(f"Skipping due to resolution: {video_url} ({width}x{height})")
            continue

        video_id = video_url.split("/")[-1]  # Use the file name from the URL
        output_path = os.path.join(output_dir, video_id)

        # Download the video if it doesn't already exist
        if not os.path.exists(output_path):
            download_video(video_url, output_path)

        # Save the caption
        captions[video_id] = caption

    # Save captions to a JSON file
    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=4)
    print(f"Saved captions to {captions_path}")


def main():
    print("Loading WebVid-10M dataset...")
    dataset = load_dataset("TempoFunk/webvid-10M", split="train[:10000]")  # Subset size

    # Download videos and captions
    download_filtered_webvid(dataset, RAW_VIDEO_PATH, CAPTIONS_PATH, min_resolution=720)


if __name__ == "__main__":
    main()
