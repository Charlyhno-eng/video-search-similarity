import os
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.extract_features(img_tensor)
        pooled = F.adaptive_avg_pool2d(features, 1)
        embedding = pooled.view(pooled.size(0), -1)
    return embedding.squeeze(0).numpy()

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
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

videos_dir = "videos_database"
embeddings_dir = "embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

for video_name in os.listdir(videos_dir):
    if not video_name.lower().endswith(".mp4"):
        continue
    video_path = os.path.join(videos_dir, video_name)
    print(f"Processing {video_name} ...")
    embedding = extract_video_embedding(video_path, frame_rate=1)  # 1 frame par seconde
    if embedding is not None:
        save_path = os.path.join(embeddings_dir, video_name + ".npy")
        np.save(save_path, embedding)
        print(f"Saved embedding for {video_name} with shape {embedding.shape}")
    else:
        print(f"No frames extracted from {video_name}")
