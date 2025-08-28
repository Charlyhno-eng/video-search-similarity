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

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos_database")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
THUMBNAILS_DIR = os.path.join(ROOT_DIR, "thumbnails")

app.mount("/videos_database", StaticFiles(directory=VIDEOS_DIR), name="videos_database")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]

# --- Extract frames from uploaded video bytes ---
def extract_frames_from_bytes(video_bytes, frame_rate=1):
    tmp_file = "temp_video.mp4"
    # Save uploaded video temporarily to disk
    with open(tmp_file, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(tmp_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * frame_rate))  # extract frames at the given rate
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # Convert BGR (OpenCV default) to RGB and store as PIL image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_idx += 1

    cap.release()
    os.remove(tmp_file)  # clean up temporary file
    return frames

# --- Extract EfficientNet embeddings from video frames ---
def extract_video_embedding_from_bytes(video_bytes, frame_rate=1):
    frames = extract_frames_from_bytes(video_bytes, frame_rate=frame_rate)
    embeddings = []

    for img in frames:
        img_tensor = transform(img).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            # Extract features from EfficientNet
            features = model.extract_features(img_tensor)
            # Global average pooling to get single vector per frame
            pooled = F.adaptive_avg_pool2d(features, 1)
            embedding = pooled.view(pooled.size(0), -1)
            embeddings.append(embedding.squeeze(0).numpy())

    if len(embeddings) == 0:
        return None

    # Average embeddings across all frames to get a single video embedding
    return np.mean(embeddings, axis=0)

# --- Load all embeddings from disk (including subfolders) ---
video_embeddings = {}
for root, _, files in os.walk(EMBEDDINGS_DIR):
    for f in files:
        if f.endswith('.npy'):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, EMBEDDINGS_DIR)  # store relative path
            video_embeddings[rel_path] = np.load(full_path)

# --- Compute cosine similarity to find top-k similar videos ---
def find_similar_videos(query_embedding, top_k=6):
    similarities = []
    for rel_path, db_embedding in video_embeddings.items():
        # compute cosine similarity between query and database embedding
        sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            db_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((rel_path, sim))
    # sort descending by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

BASE_VIDEO_URL = "http://localhost:8000/videos_database/"
BASE_THUMB_URL = "http://localhost:8000/thumbnails/"

# --- FastAPI endpoint to upload a video and find similar ones ---
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = extract_video_embedding_from_bytes(contents, frame_rate=1)

    if embedding is None:
        return {"error": "No frames could be extracted from the video."}

    similar_videos = find_similar_videos(embedding, top_k=6)

    results = []
    for rel_path, sim in similar_videos:
        # Remove '.npy' extension to get original video name
        video_name = rel_path[:-4]

        # URL for the video
        video_url = BASE_VIDEO_URL + video_name.replace("\\", "/")

        # Remove video file extension for thumbnail URL
        filename_no_ext = os.path.basename(video_name)
        for ext in VIDEO_EXTENSIONS:
            if filename_no_ext.lower().endswith(ext):
                filename_no_ext = filename_no_ext[:-len(ext)]
                break

        subfolder = os.path.dirname(video_name).replace("\\", "/")

        # Build thumbnail URL, handle subfolder or root
        thumbnail_url = f"{BASE_THUMB_URL}{subfolder}/{filename_no_ext}.jpg" if subfolder else f"{BASE_THUMB_URL}{filename_no_ext}.jpg"

        results.append({
            "filename": os.path.basename(video_name),
            "similarity": float(sim),
            "url": video_url,
            "thumbnail_url": thumbnail_url,
            "subfolder": subfolder or "root"
        })

    # Thumbnail URL for uploaded video (optional, may not exist)
    uploaded_base_name, _ = os.path.splitext(file.filename)
    uploaded_thumbnail_url = f"{BASE_THUMB_URL}{uploaded_base_name}.jpg"

    return {
        "filename": file.filename,
        "embedding": embedding.tolist(),
        "thumbnail_url": uploaded_thumbnail_url,
        "similar_videos": results,
        "message": "Video received, embedding extracted, similar videos found"
    }
