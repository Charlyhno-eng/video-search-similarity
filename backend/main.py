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
import chromadb
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
VIDEOS_DIR = os.path.join(ROOT_DIR, "db_videos")
THUMBNAILS_DIR = os.path.join(ROOT_DIR, "db_thumbnails")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app.mount("/db_videos", StaticFiles(directory=VIDEOS_DIR), name="db_videos")
app.mount("/db_thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="db_thumbnails")

BASE_VIDEO_URL = "http://localhost:8000/db_videos/"
BASE_THUMB_URL = "http://localhost:8000/db_thumbnails/"

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ChromaDB client ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="videos")

# --- ThreadPoolExecutor for heavy processing ---
executor = ThreadPoolExecutor(max_workers=4)

# --- Helpers ---
def extract_frames_from_bytes(video_bytes, n_frames=10):
    """
    Extract a limited number of frames from a video given as bytes.

    Parameters:
        video_bytes (bytes): Raw video data.
        n_frames (int): Number of frames to extract evenly across the video.

    Returns:
        List[PIL.Image.Image]: A list of extracted frames as PIL images.
    """
    tmp_file = "temp_video.mp4"
    with open(tmp_file, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(tmp_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        os.remove(tmp_file)
        return []

    frame_indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    os.remove(tmp_file)
    return frames


def extract_video_embedding_from_bytes_sync(video_bytes):
    """
    Synchronously extract a video embedding from raw video bytes using EfficientNet.

    Parameters:
        video_bytes (bytes): Raw video data.

    Returns:
        np.ndarray or None: The mean embedding vector of selected frames, or None if no frames extracted.
    """
    frames = extract_frames_from_bytes(video_bytes, n_frames=5)
    if not frames:
        return None
    img_tensors = torch.stack([transform(img) for img in frames]).to(device)
    with torch.no_grad():
        features = model.extract_features(img_tensors)
        pooled = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        embeddings = pooled.cpu().numpy()
    return np.mean(embeddings, axis=0)


async def extract_video_embedding_from_bytes(video_bytes):
    """
    Asynchronously extract a video embedding using a thread pool to avoid blocking the event loop.

    Parameters:
        video_bytes (bytes): Raw video data.

    Returns:
        np.ndarray or None: The mean embedding vector of selected frames, or None if no frames extracted.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, extract_video_embedding_from_bytes_sync, video_bytes)


def generate_thumbnail(video_path, thumb_path):
    """
    Generate a thumbnail from the middle frame of a video.

    Parameters:
        video_path (str): Path to the video file.
        thumb_path (str): Path to save the generated thumbnail image.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    idx = 0
    ret = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx == mid_frame_idx:
            cv2.imwrite(thumb_path, frame)
            break
        idx += 1
    cap.release()


def distance_to_similarity(distance):
    """
    Convert a distance metric to a similarity percentage.

    Parameters:
        distance (float): Distance between embeddings.

    Returns:
        float: Similarity percentage (0-100).
    """
    return float(100 * (1 / (1 + distance)))


# --- Endpoints ---
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()

    embedding = await extract_video_embedding_from_bytes(contents)
    if embedding is None:
        return {"error": "No frames could be extracted"}

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=6
    )

    similar_videos = []
    for i, vid_id in enumerate(results['ids'][0]):
        metadata = results['metadatas'][0][i]
        filename = os.path.basename(metadata['video_path'])
        label = metadata.get("class", "unknown")
        thumb_url = BASE_THUMB_URL + f"{label}/{os.path.splitext(filename)[0]}.jpg"

        distance = results['distances'][0][i] if 'distances' in results else None
        similarity_percent = distance_to_similarity(distance) if distance is not None else None

        similar_videos.append({
            "filename": filename,
            "similarity": similarity_percent,
            "url": metadata['video_path'],
            "thumbnail_url": thumb_url,
            "subfolder": label
        })

    uploaded_base_name = os.path.splitext(file.filename)[0]
    uploaded_thumb_url = BASE_THUMB_URL + uploaded_base_name + ".jpg"

    return {
        "filename": file.filename,
        "embedding": embedding.tolist(),
        "thumbnail_url": uploaded_thumb_url,
        "similar_videos": similar_videos,
        "message": "Video received, embedding extracted, similar videos found"
    }

@app.get("/get-classes/")
async def get_classes():
    classes = [d for d in os.listdir(VIDEOS_DIR) if os.path.isdir(os.path.join(VIDEOS_DIR, d))]
    return {"classes": classes}

@app.post("/create-class/")
async def create_class(className: str = Body(..., embed=True)):
    formatting_class_name = "".join([c.lower() if c.isalnum() else "_" for c in className])
    video_dir = os.path.join(VIDEOS_DIR, formatting_class_name)
    thumb_dir = os.path.join(THUMBNAILS_DIR, formatting_class_name)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)
    return {"message": f"Class '{formatting_class_name}' created successfully"}

@app.post("/add-video/")
async def add_video(file: UploadFile = File(...), label: str = Form(...)):
    video_subdir = os.path.join(VIDEOS_DIR, label)
    thumb_subdir = os.path.join(THUMBNAILS_DIR, label)
    os.makedirs(video_subdir, exist_ok=True)
    os.makedirs(thumb_subdir, exist_ok=True)

    video_path = os.path.join(video_subdir, file.filename)
    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    embedding = await extract_video_embedding_from_bytes(contents)
    if embedding is None:
        return {"error": "Impossible to extract embedding"}

    thumb_path = os.path.join(thumb_subdir, os.path.splitext(file.filename)[0] + ".jpg")
    generate_thumbnail(video_path, thumb_path)

    video_id = f"{label}_{os.path.splitext(file.filename)[0]}"
    collection.add(
        ids=[video_id],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "class": label,
            "video_path": BASE_VIDEO_URL + f"{label}/{file.filename}"
        }]
    )

    return {
        "message": f"Video {file.filename} added to class {label}",
        "video_path": video_path,
        "thumbnail_path": thumb_path
    }
