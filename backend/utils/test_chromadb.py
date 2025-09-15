import numpy as np
import chromadb
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Initialize Chroma client ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("videos")

# --- 2. Load EfficientNet-B4 (PyTorch) ---
model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Function to extract embedding from video ---
def extract_embedding_from_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
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

    embeddings = []
    for img in frames:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model.extract_features(img_tensor)
            pooled = F.adaptive_avg_pool2d(features, 1)
            embeddings.append(pooled.view(pooled.size(0), -1).squeeze(0).numpy())
    if not embeddings:
        raise ValueError(f"No frames could be extracted from {video_path}")
    return np.mean(embeddings, axis=0)

# --- 4. Input video ---
input_video_path = "db_videos/calm_water/649i810.avi"
input_embedding = extract_embedding_from_video(input_video_path)

# --- 5. Query ChromaDB ---
results = collection.query(
    query_embeddings=[input_embedding.tolist()],
    n_results=6
)

# --- 6. Display results ---
print("Top 6 similar videos:")
for i, vid_id in enumerate(results['ids'][0]):
    metadata = results['metadatas'][0][i]
    print(f"{i+1}. ID: {vid_id}")
    print(f"   Class: {metadata['class']}")
    print(f"   Video path: {metadata['video_path']}")
    print(f"   Thumbnail path: {metadata['thumbnail_path']}")
    print("-" * 40)

if 'distances' in results:
    print("Distances:", results['distances'][0])
