# video-search-similarity

This project was developed as part of a research internship at Politehnica University of Timișoara in the field of computer vision.
Its goal is to detect and retrieve video patterns of similarity in ocean, river, and stream footage.

The application enables video search by similarity using **EfficientNet-B4 embeddings**.
The frontend is built with **Next.js** and **MUI**, while the backend is implemented in **Python (FastAPI)**.

---

## Intallation

### 1. Set up the video database
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir videos_database
```

The application supports the following video formats: .mp4, .avi, and .mov.

First, create a videos_database folder inside the backend directory.
Then, add subfolders named after the classes of interest, and place your videos inside these subfolders.
If a class name contains multiple words, please use underscores (_) instead of spaces.

Example folder structure:

```bash
└── videos_database
    ├── beach_and_waves
    ├── big_rocks
    ├── blue_water
    ├── boat
    ├── cliff_water
    ├── dirty_water
    ├── grass
    ├── high_grass
    ├── human
    ├── little_rock_in_water
    ├── mountain_and_water
    ├── rock_in_water
    ├── tree_in_back
    └── water_tree_reflect
```

### 2. Generate embeddings

From the backend directory, run:

```bash
python utils/process_videos.py
```

This script will generate embeddings for each video as well as thumbnails for display in the application.
Execution time may vary depending on the size of your video database.

### 3. Start the backend server

Still in the backend directory, run:

```bash
uvicorn main:app --reload
```

### 4. Launch the frontend application

```bash
cd ../frontend
npm install
npm run dev
```

---

## Demonstration

For performance reasons, the full video is not displayed in the web application.
Instead, the application shows the **video filename** along with its **first frame (thumbnail)**, similar to how YouTube displays video previews.
Additionally, the **similarity percentage** between the query image and the images in the database is shown.

For testing purposes, I used videos with relatively distinct content:
- a video of a cliff,
- a video of rocks in turquoise water,
- and a video of a forest by a river.

![First test](public/test1.png)

![Second test](public/test2.png)

![Third test](public/test3.png)
