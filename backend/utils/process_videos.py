import os
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import chromadb

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(ROOT_DIR, "db_videos")
THUMBNAILS_DIR = os.path.join(ROOT_DIR, "db_thumbnails")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")

os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- Model ---
model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ChromaDB ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="videos",
    metadata={"hnsw:space": "cosine"}
)

# --- Helpers ---
def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from a video file at a specified frame rate.

    Parameters:
        video_path (str): Path to the video file.
        frame_rate (int or float): Number of frames per second to extract.

    Returns:
        List[PIL.Image.Image]: List of extracted frames as PIL images.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return []
    interval = max(1, int(fps * frame_rate))
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_idx += 1
    cap.release()
    return frames


def extract_video_embedding(video_path, frame_rate=1):
    """
    Extract a normalized video embedding by processing sampled frames through EfficientNet.

    Parameters:
        video_path (str): Path to the video file.
        frame_rate (int or float): Frame extraction rate for embedding calculation.

    Returns:
        np.ndarray or None: Normalized mean embedding vector, or None if no frames extracted.
    """
    frames = extract_frames(video_path, frame_rate=frame_rate)
    embeddings = []
    for img in frames:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model.extract_features(img_tensor)
            pooled = F.adaptive_avg_pool2d(features, 1)
            emb = pooled.view(pooled.size(0), -1).squeeze(0).numpy()
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    return mean_emb


def generate_thumbnail(video_path, thumbnail_path):
    """
    Generate a thumbnail image from the first frame of a video.

    Parameters:
        video_path (str): Path to the video file.
        thumbnail_path (str): Path where the thumbnail image will be saved.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        cv2.imwrite(thumbnail_path, frame)


# --- Migration to ChromaDB ---
if __name__ == "__main__":
    for root, _, files in os.walk(VIDEOS_DIR):
        for video_name in files:
            if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                continue

            video_path = os.path.join(root, video_name)
            rel_path = os.path.relpath(video_path, VIDEOS_DIR)
            label = os.path.dirname(rel_path).replace("\\", "/") or "root"

            # Thumbnail
            thumb_path = os.path.join(THUMBNAILS_DIR, os.path.splitext(rel_path)[0] + ".jpg")
            generate_thumbnail(video_path, thumb_path)

            # Embedding
            embedding = extract_video_embedding(video_path, frame_rate=1)
            if embedding is None:
                print(f"Unable to extract embedding for {video_path}")
                continue

            # Add to ChromaDB
            video_id = f"{label}_{os.path.splitext(video_name)[0]}"
            collection.add(
                ids=[video_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "class": label,
                    "video_path": f"http://localhost:8000/db_videos/{label}/{video_name}",
                    "thumbnail_path": f"http://localhost:8000/db_thumbnails/{label}/{os.path.splitext(video_name)[0]}.jpg"
                }]
            )
            print(f"Video added : {video_id}")

    print("All videos have been processed and embeddings added to ChromaDB!")
