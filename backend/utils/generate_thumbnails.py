import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos_database")
THUMBNAILS_DIR = os.path.join(BASE_DIR, "thumbnails")

os.makedirs(THUMBNAILS_DIR, exist_ok=True)

def generate_thumbnail(video_path, thumbnail_path):
    """Extrait la première frame et la sauvegarde en JPEG"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(thumbnail_path, frame)
        print(f"Thumbnail saved: {thumbnail_path}")
    else:
        print(f"Failed to extract frame from {video_path}")

if __name__ == "__main__":
    for video_name in os.listdir(VIDEOS_DIR):
        if not video_name.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEOS_DIR, video_name)
        thumbnail_path = os.path.join(THUMBNAILS_DIR, video_name.replace(".mp4", ".jpg"))
        generate_thumbnail(video_path, thumbnail_path)
