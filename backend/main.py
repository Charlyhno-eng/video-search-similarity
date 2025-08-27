import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
videos_path = os.path.join(BASE_DIR, "videos_database")
embeddings_dir = os.path.join(BASE_DIR, "embeddings")
thumbnails_path = os.path.join(BASE_DIR, "thumbnails")

# Serve static files
app.mount("/videos_database", StaticFiles(directory=videos_path), name="videos_database")
app.mount("/thumbnails", StaticFiles(directory=thumbnails_path), name="thumbnails")

# EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper: extract frames from video bytes
def extract_frames_from_bytes(video_bytes, frame_rate=1):
    tmp_file = "temp_video.mp4"
    with open(tmp_file, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(tmp_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    os.remove(tmp_file)
    return frames

# Helper: extract video embedding
def extract_video_embedding_from_bytes(video_bytes, frame_rate=1):
    frames = extract_frames_from_bytes(video_bytes, frame_rate=frame_rate)
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

# Load precomputed embeddings
video_embeddings = {}
for f in os.listdir(embeddings_dir):
    if f.endswith('.npy'):
        path = os.path.join(embeddings_dir, f)
        video_embeddings[f[:-4]] = np.load(path)

# Similarity search
def find_similar_videos(query_embedding, top_k=6):
    similarities = []
    for filename, db_embedding in video_embeddings.items():
        sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            db_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((filename, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Base URL for videos
BASE_VIDEO_URL = "http://localhost:8000/videos_database/"

# Upload endpoint
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = extract_video_embedding_from_bytes(contents, frame_rate=1)

    if embedding is None:
        return {"error": "No frames could be extracted from the video."}

    similar_videos = find_similar_videos(embedding, top_k=6)

    results = []
    for fname, sim in similar_videos:
        video_url = BASE_VIDEO_URL + fname
        base_name, _ = os.path.splitext(fname)  # support mp4, mov, avi
        thumbnail_url = f"http://localhost:8000/thumbnails/{base_name}.jpg"
        results.append({
            "filename": fname,
            "similarity": float(sim),
            "url": video_url,
            "thumbnail_url": thumbnail_url
        })

    uploaded_base_name, _ = os.path.splitext(file.filename)  # support mp4, mov, avi
    uploaded_thumbnail_url = f"http://localhost:8000/thumbnails/{uploaded_base_name}.jpg"

    return {
        "filename": file.filename,
        "embedding": embedding.tolist(),
        "thumbnail_url": uploaded_thumbnail_url,
        "similar_videos": results,
        "message": "Video received, embedding extracted, similar videos found"
    }
