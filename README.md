# **video-search-similarity**

This project was developed as part of a research internship at Politehnica University of Timișoara in the field of computer vision.
Its goal is to detect and retrieve video patterns of similarity in ocean, river, and stream footage.

The application enables video search by similarity using **EfficientNet-B4 embeddings** and the **cosine similarity** method for comparison.
A **ChromaDB vector database** is used to store and query embeddings efficiently.
The frontend is built with **Next.js** and **MUI**, while the backend is implemented in **Python (FastAPI)**.

- If you would like to **install the application**, click [here](#intallation).
- If you want to **see a demonstration**, click [here](#demonstration).
- If you are interested in **explanations and technology choices**, click [here](#explanations-and-comparisons).
- If you want to **know the sources I used**, click [here](#my-resources).

---

## Intallation

### 1. Set up the video database

Regarding the installation, you have two options. If you are on Linux Mint or a Linux with a distribution similar to Linux Mint, I have a script "installation_script.sh" at the root of the project. This script installs everything from A to Z, it installs node, python, then installs the python libraries necessary for the project and does the same for nextjs.<br />

The other option is to follow these steps :
- Clone the Github repository
- Go to the repository:
```bash
cd video-search-similarity
```

### 3. Start the backend server

Still in the backend directory, run:

```bash
cd backend
python -m venv venv && source venv/bin/activate
uvicorn main:app --reload
```

### 4. Launch the frontend application

```bash
cd frontend
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

![First example](public/example1.png)

![Second example](public/example2.png)

![Third example](public/example3.png)

---

## Explanations and comparisons

### What is an embedding?

An embedding is basically a way to turn a complex piece of data, like an image or a video frame, into a vector of numbers.
Instead of storing all the raw pixels (which would be huge and not very useful for comparison), a deep learning model like EfficientNet processes the frame and outputs a feature vector. This vector is called an **embedding**.

You can think of it as a kind of "summary" of the image: it keeps the most important information about what is in the frame, but in a compact numerical form. Two images that look similar to humans will usually have embeddings that are close to each other in this vector space.

The embedding does not directly store things like “this is water” or “this is a rock” in plain text. Instead, it encodes **patterns and visual properties** that the neural network has learned to recognize. Here are some of the main aspects that are captured in the vector:

1. **Colors and textures**
   - The model detects overall color distributions (e.g., turquoise water vs. dark muddy water).
   - It also captures texture information, like the difference between smooth water, rough waves, or sharp rock surfaces.

2. **Shapes and edges**
   - Embeddings contain signals about the geometry of objects. For example, cliffs have strong vertical edges, while trees have more irregular shapes.
   - This helps the model understand the structure of what’s in the image, beyond just color.

3. **Objects and patterns**
   - Thanks to training on large datasets, the model has learned to recognize high-level concepts like “boat,” “tree,” or “person.”
   - Even if it doesn’t give a label, these object features are reflected in the vector.

4. **Spatial relationships**
   - Embeddings also keep some information about how objects are arranged. For instance, water below a cliff looks different from water next to grass.
   - It’s not as precise as exact coordinates, but enough to distinguish scenes.

5. **Contextual cues**
   - Sometimes the “feeling” of a scene is captured: a calm blue ocean vs. a stormy grey sea will produce different embeddings because of the global combination of colors, textures, and shapes.

When you compare two embeddings, you’re basically comparing these abstract features.
If two videos share similar colors, textures, objects, and patterns, their embeddings will be close to each other in the vector space.
That’s why you can measure similarity with something like **cosine similarity** or **Euclidean distance** between embeddings: smaller distance = more similar videos.

So, in short, the embedding is a numerical fingerprint of each video frame. By looking at these fingerprints instead of raw pixels, the application can quickly figure out which videos are most alike.

---

### Calculation of distance between vectors

- #### **Euclidean Distance (L2 norm)**

Euclidean distance is one of the most common methods for measuring similarity between vectors.
It calculates the straight-line distance between two points in vector space:

$$
d(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

It considers both magnitude and direction of the vectors, making it intuitive but sensitive to scale.

- #### **Cosine Similarity / Cosine Distance**

Cosine similarity measures the angle between two vectors, ignoring their magnitude.
It is often converted into a distance:

$$
\text{cosine similarity}(x,y) = \frac{x \cdot y}{\|x\| \, \|y\|}, \quad
d(x,y) = 1 - \text{cosine similarity}(x,y)
$$

This metric is widely used in embeddings since it focuses on orientation rather than vector length.

- #### **Hamming Distance**

The Hamming distance counts the number of positions at which two vectors (or strings) differ:

$$
d(x,y) = \sum_{i=1}^{n} [x_i \neq y_i]
$$

It is mainly used for binary vectors or categorical data.

- #### **Manhattan Distance (L1 norm)**

Also known as the "taxicab" distance, it sums the absolute differences between coordinates:

$$
d(x,y) = \sum_{i=1}^{n} |x_i - y_i|
$$

It is less sensitive to outliers than Euclidean distance.

- #### **Chebyshev Distance (L∞ norm)**

Chebyshev distance only considers the largest difference across any dimension:

$$
d(x,y) = \max_i |x_i - y_i|
$$

It is useful when the greatest single difference dominates similarity.

- #### **Minkowski Distance**

A generalization of L1 and L2 distances, parameterized by $p$:

$$
d(x,y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

For $p=1$ it becomes Manhattan distance, and for $p=2$ it becomes Euclidean distance.

- #### **Jaccard Distance**

The Jaccard distance compares two sets by measuring their overlap:

$$
d(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}
$$

It is useful for binary data or sets of items.

- #### **Sørensen–Dice Distance**

A variation of Jaccard, the Dice distance doubles the weight of the intersection:

$$
d(A,B) = 1 - \frac{2 |A \cap B|}{|A| + |B|}
$$

It is often used in text and biological sequence comparison.

---

### Why I used EfficientNet ?

| Category                  | Model / Architecture          | Typical Enterprise Use                                  | Strengths                                       | Weaknesses                                                             |
| ------------------------- | ----------------------------- | ------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| **Classic CNNs**          | ResNet, VGG, Inception        | Embedding extraction for short to medium videos         | Stable, robust, well-documented                 | Heavy, may require GPUs for large datasets                             |
| **Lightweight models**    | MobileNet, EfficientNet       | Web or mobile applications, rapid prototyping           | Lightweight, fast, easy to deploy               | Slightly lower accuracy than heavy models                              |
| **Vision Transformers**   | TimeSformer, VideoSwin, ViViT | Advanced temporal and semantic similarity search        | Excellent accuracy, captures video dynamics     | Very heavy, requires powerful GPU, hard to deploy in browser           |
| **Multimodal / Semantic** | CLIP, OpenCLIP                | Semantic search (text ↔ video), advanced recommendation | Very powerful for understanding overall content | Computationally expensive, needs GPU, challenging for very long videos |
| **3D CNNs**               | C3D, I3D                      | Action or motion analysis, surveillance                 | Captures temporal dynamics of videos            | Very heavy, typically requires GPU servers                             |

EfficientNet is known for its lightweight and efficient performance, making it suitable for applications requiring fast execution, particularly in web browsers. It offers a good balance between accuracy and performance, particularly for short videos. Comparative studies, such as the one published in the International Journal of Scientific Research in Engineering and Management, highlight its efficiency in terms of computation and accuracy.

---

### Why I used ChromaDB ?

---

## My resources

- [1]. TileDB Academy, “Distance Metrics” _Web site_. [Online]. Available: [https://cloud.tiledb.com/academy/structure/ai-ml/vector-search/foundation/key-concepts/distance-metrics/](https://cloud.tiledb.com/academy/structure/ai-ml/vector-search/foundation/key-concepts/distance-metrics/)

- [2]. LeMarson TV, “Les bases de données vectorielles” _Youtube video_, 2025. [Online]. Available: [https://www.youtube.com/watch?v=DB1vFroa_es&list=LL&index=6](https://www.youtube.com/watch?v=DB1vFroa_es&list=LL&index=6)

- [3]. ResearchGate, “Comparative Study of Image Classification Models Using Deep Learning: MobileNet, ResNet, and EfficientNet” _PDF_, 2025. [Online]. Available: [https://www.researchgate.net/publication/391845914_Comparative_Study_of_Image_Classification_Models_Using_Deep_Learning_MobileNet_ResNet_and_EfficientNet](https://www.researchgate.net/publication/391845914_Comparative_Study_of_Image_Classification_Models_Using_Deep_Learning_MobileNet_ResNet_and_EfficientNet)
