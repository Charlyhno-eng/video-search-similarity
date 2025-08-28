import os
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos_database")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
THUMBNAILS_DIR = os.path.join(ROOT_DIR, "thumbnails")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return []
    interval = int(fps * frame_rate)
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
    frames = extract_frames(video_path, frame_rate=frame_rate)
    embeddings = []
    for img in frames:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model.extract_features(img_tensor)
            pooled = F.adaptive_avg_pool2d(features, 1)
            embedding = pooled.view(pooled.size(0), -1)
            embeddings.append(embedding.squeeze(0).numpy())
    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)

def generate_thumbnail(video_path, thumbnail_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(thumbnail_path, frame)

if __name__ == "__main__":
    for root, _, files in os.walk(VIDEOS_DIR):
        for video_name in files:
            if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(root, video_name)
            rel_path = os.path.relpath(video_path, VIDEOS_DIR)

            embedding_path = os.path.join(EMBEDDINGS_DIR, rel_path + ".npy")
            thumbnail_path = os.path.join(THUMBNAILS_DIR, rel_path)
            thumbnail_path = os.path.splitext(thumbnail_path)[0] + ".jpg"

            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)

            embedding = extract_video_embedding(video_path, frame_rate=1)
            if embedding is not None:
                np.save(embedding_path, embedding)

            generate_thumbnail(video_path, thumbnail_path)
