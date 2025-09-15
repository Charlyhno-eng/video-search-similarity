import os
import numpy as np
import chromadb

# 1. Initialize Chroma client (persistent, stored locally)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="videos")

# 2. Source folders
DB_EMBEDDINGS = "db_embeddings"
DB_VIDEOS = "db_videos"
DB_THUMBNAILS = "db_thumbnails"

# 3. Possible video extensions
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

def find_matching_video(base_path, base_name, extensions):
    """
    Look for a video file that matches base_name with one of the given extensions.
    Example: base_name="video1" → check video1.mp4, video1.avi, etc.
    """
    for ext in extensions:
        candidate = os.path.join(base_path, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# 4. Migration
for class_name in os.listdir(DB_EMBEDDINGS):
    class_path = os.path.join(DB_EMBEDDINGS, class_name)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if not file.endswith(".npy"):
            continue

        # Clean base name: remove ".npy" and possible video extension before it
        base_name = file.replace(".npy", "")
        for ext in VIDEO_EXTENSIONS:
            if base_name.endswith(ext):
                base_name = base_name.replace(ext, "")
                break

        embedding_path = os.path.join(class_path, file)

        # Find corresponding video and thumbnail
        video_path = find_matching_video(os.path.join(DB_VIDEOS, class_name), base_name, VIDEO_EXTENSIONS)
        thumbnail_path = os.path.join(DB_THUMBNAILS, class_name, base_name + ".jpg")

        if video_path is None or not os.path.exists(thumbnail_path):
            print(f"⚠️ Skipping {embedding_path}: no matching video or thumbnail found.")
            continue

        # Load embedding
        embedding = np.load(embedding_path)

        # Unique ID for Chroma
        video_id = f"{class_name}_{base_name}"

        # Add entry to Chroma
        collection.add(
            ids=[video_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "class": class_name,
                "video_path": video_path,
                "thumbnail_path": thumbnail_path
            }]
        )

        print(f"Added {video_id} to ChromaDB.")

print("Migration finished successfully!")
