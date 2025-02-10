import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import json

# Paths
FRAMES_DIR = "../data/frames/"
CAPTIONS_PATH = "../data/captions/processed_captions.json"
EMBEDDINGS_DIR = "../data/embeddings/"

# New Subfolders
FRAME_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, "frames/")
TEXT_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, "text/")

# Ensure directories exist
os.makedirs(FRAME_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(TEXT_EMBEDDINGS_DIR, exist_ok=True)

# Load CLIP Model and Tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def extract_frame_embeddings(frame_path):
    """Extract embeddings for a single frame."""
    image = Image.open(frame_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)

    # Validate shape
    assert embeddings.shape[1] == 512, f"Frame embedding shape mismatch: {embeddings.shape}"

    print(f"✔️ Frame embedding extracted for {frame_path}, Shape: {embeddings.shape}")  # Debug print

    return embeddings.cpu().numpy()


def process_frame_embeddings():
    """Extract embeddings for all frames and save them separately."""
    frame_files = sorted(os.listdir(FRAMES_DIR))
    print(f"📸 Found {len(frame_files)} frames in {FRAMES_DIR}")

    for frame_file in tqdm(frame_files, desc="Processing frame embeddings"):
        frame_path = os.path.join(FRAMES_DIR, frame_file)

        # Skip directories if any exist (just in case)
        if os.path.isdir(frame_path):
            print(f"⚠️ Skipping directory: {frame_path}")
            continue

        embeddings = extract_frame_embeddings(frame_path)

        if embeddings is None:
            print(f"⚠️ No embeddings extracted for {frame_path}")
        else:
            # Save embeddings with same frame filename
            output_path = os.path.join(FRAME_EMBEDDINGS_DIR, f"{frame_file}.npy")
            np.save(output_path, embeddings)
            print(f"✅ Saved frame embedding for {frame_file} at {output_path}")


def extract_text_embeddings(caption):
    """Extract embeddings for a text caption using CLIPModel and validate shape."""
    inputs = clip_processor.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)

    # Validate shape: Expecting [batch_size, 512] (batch_size=1 here)
    assert embeddings.shape[1] == 512, f"Text embedding shape mismatch: {embeddings.shape}"

    return embeddings.cpu().numpy()


def process_text_embeddings():
    """Extract embeddings for all captions and save them separately."""
    with open(CAPTIONS_PATH, "r") as f:
        captions = json.load(f)

    for video_id, caption in tqdm(captions.items(), desc="Processing captions"):
        embeddings = extract_text_embeddings(caption)
        output_path = os.path.join(TEXT_EMBEDDINGS_DIR, f"{video_id}.npy")
        np.save(output_path, embeddings)
        print(f"Saved text embeddings for {video_id} at {output_path}")


def main():
    print("Extracting frame embeddings...")
    process_frame_embeddings()
    print("Extracting text embeddings...")
    process_text_embeddings()


if __name__ == "__main__":
    main()
