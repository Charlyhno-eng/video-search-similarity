import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos_database")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
THUMBNAILS_DIR = os.path.join(ROOT_DIR, "thumbnails")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

app.mount("/videos_database", StaticFiles(directory=VIDEOS_DIR), name="videos_database")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
BASE_VIDEO_URL = "http://localhost:8000/videos_database/"
BASE_THUMB_URL = "http://localhost:8000/thumbnails/"

# --- Model ---
model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helpers ---
def extract_frames_from_bytes(video_bytes, frame_rate=1):
    tmp_file = "temp_video.mp4"
    with open(tmp_file, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(tmp_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * frame_rate))
    frames, frame_idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += 1
    cap.release()
    os.remove(tmp_file)
    return frames

def extract_video_embedding_from_bytes(video_bytes, frame_rate=1):
    frames = extract_frames_from_bytes(video_bytes, frame_rate)
    embeddings = []
    for img in frames:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model.extract_features(img_tensor)
            pooled = F.adaptive_avg_pool2d(features, 1)
            embeddings.append(pooled.view(pooled.size(0), -1).squeeze(0).numpy())
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)

# --- Load existing embeddings ---
video_embeddings = {}
for root, _, files in os.walk(EMBEDDINGS_DIR):
    for f in files:
        if f.endswith('.npy'):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, EMBEDDINGS_DIR)
            video_embeddings[rel_path] = np.load(full_path)

def find_similar_videos(query_embedding, top_k=6):
    sims = [(rel_path, cosine_similarity(query_embedding.reshape(1, -1), db_emb.reshape(1, -1))[0][0])
            for rel_path, db_emb in video_embeddings.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# --- Endpoints ---
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = extract_video_embedding_from_bytes(contents, frame_rate=1)
    if embedding is None:
        return {"error": "No frames could be extracted"}

    similar_videos = find_similar_videos(embedding, top_k=6)
    results = []
    for rel_path, sim in similar_videos:
        video_name = rel_path[:-4]
        video_url = BASE_VIDEO_URL + video_name.replace("\\", "/")
        filename_no_ext = os.path.splitext(os.path.basename(video_name))[0]
        subfolder = os.path.dirname(video_name).replace("\\", "/")
        thumbnail_url = f"{BASE_THUMB_URL}{subfolder}/{filename_no_ext}.jpg" if subfolder else f"{BASE_THUMB_URL}{filename_no_ext}.jpg"
        results.append({
            "filename": os.path.basename(video_name),
            "similarity": float(sim),
            "url": video_url,
            "thumbnail_url": thumbnail_url,
            "subfolder": subfolder or "root"
        })

    uploaded_base_name, _ = os.path.splitext(file.filename)
    uploaded_thumbnail_url = f"{BASE_THUMB_URL}{uploaded_base_name}.jpg"

    return {
        "filename": file.filename,
        "embedding": embedding.tolist(),
        "thumbnail_url": uploaded_thumbnail_url,
        "similar_videos": results,
        "message": "Video received, embedding extracted, similar videos found"
    }

@app.get("/get-classes/")
async def get_classes():
    try:
        classes = [d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d))]
        return {"classes": classes}
    except Exception as e:
        return {"classes": [], "error": str(e)}

@app.post("/create-class/")
async def create_class(className: str = Body(..., embed=True)):
    formatting_class_name = "".join([c.lower() if c.isalnum() else "_" for c in className])
    video_dir = os.path.join(VIDEOS_DIR, formatting_class_name)
    emb_dir = os.path.join(EMBEDDINGS_DIR, formatting_class_name)
    thumb_dir = os.path.join(THUMBNAILS_DIR, formatting_class_name)

    try:
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(thumb_dir, exist_ok=True)
        return {"message": f"Class '{formatting_class_name}' created successfully"}
    except Exception as e:
        return {"message": str(e)}

@app.post("/add-video/")
async def add_video(file: UploadFile = File(...), label: str = Form(...)):
    video_subdir = os.path.join(VIDEOS_DIR, label)
    emb_subdir = os.path.join(EMBEDDINGS_DIR, label)
    thumb_subdir = os.path.join(THUMBNAILS_DIR, label)
    os.makedirs(video_subdir, exist_ok=True)
    os.makedirs(emb_subdir, exist_ok=True)
    os.makedirs(thumb_subdir, exist_ok=True)

    video_path = os.path.join(video_subdir, file.filename)
    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    embedding = extract_video_embedding_from_bytes(contents, frame_rate=1)
    if embedding is None:
        return {"error": "Impossible to extract embedding"}

    emb_path = os.path.join(emb_subdir, file.filename + ".npy")
    np.save(emb_path, embedding)

    thumb_path = os.path.join(thumb_subdir, os.path.splitext(file.filename)[0] + ".jpg")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(thumb_path, frame)

    rel_path = os.path.relpath(emb_path, EMBEDDINGS_DIR)
    video_embeddings[rel_path] = embedding

    return {
        "message": f"Video {file.filename} added to class {label}",
        "video_path": video_path,
        "embedding_path": emb_path,
        "thumbnail_path": thumb_path
    }
