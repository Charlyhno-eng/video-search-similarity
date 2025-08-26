import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(BASE_DIR, "videos_database")
app.mount("/videos_database", StaticFiles(directory=images_path), name="videos_database")

model = EfficientNet.from_pretrained('efficientnet-b4')
model.eval()

# Image preprocessing for EfficientNet-B4 (expected size: 380x380)
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding(image: Image.Image):
    """Extract a 1792-dim embedding from image using EfficientNet-B4."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.extract_features(img_tensor)
        pooled = F.adaptive_avg_pool2d(features, 1)
        embedding = pooled.view(pooled.size(0), -1)
    return embedding.squeeze(0).numpy()

# Load existing embeddings
embeddings_dir = 'embeddings'
image_embeddings = {}
for f in os.listdir(embeddings_dir):
    if f.endswith('.npy'):
        path = os.path.join(embeddings_dir, f)
        image_embeddings[f[:-4]] = np.load(path)

def find_similar_images(query_embedding, top_k=6):
    similarities = []
    for filename, db_embedding in image_embeddings.items():
        sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            db_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((filename, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

BASE_IMAGE_URL = "http://localhost:8000/videos_database/"

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    embedding = extract_embedding(image)

    similar_images = find_similar_images(embedding, top_k=6)

    width, height = image.size

    results = []
    for fname, sim in similar_images:
        url = BASE_IMAGE_URL + fname
        results.append({"filename": fname, "similarity": float(sim), "url": url})

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "embedding": embedding.tolist(),
        "similar_images": results,
        "message": "Image received, embedding extracted, similar images found"
    }
