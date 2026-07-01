import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
from ultralytics import YOLO

# 1. Load your YOLO26 detector and a separate embedding model
# (Using yolo26 for cropping and a standard classification model for features)
detector = YOLO("yolo26n.pt")
embedder = YOLO("yolo26n-cls.pt")


def get_crops_and_embeddings(img_path):
    """Detects objects, crops them, and extracts visual embeddings."""
    img = cv2.imread(img_path)
    results = detector(img)[0]  # YOLO26 End-to-End inference (NMS-free)

    embeddings = []
    boxes = []

    # Iterate over every detected object bounding box
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].cpu().item())

        # Crop the object from the image array
        crop = img[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        if crop.size == 0:
            continue

        # Extract vector embedding from the cropped object patch
        with torch.no_grad():
            # .embed() maps the image crop to a clean feature vector
            vector = embedder.embed(crop)

        embeddings.append(vector.squeeze(0))
        boxes.append({"coords": xyxy, "class": cls_id})

    return embeddings, boxes


# 2. Process both distinct view perspectives
embeds_a, boxes_a = get_crops_and_embeddings("view_angle_A.jpg")
embeds_b, boxes_b = get_crops_and_embeddings("view_angle_B.jpg")

# 3. Construct the Distance/Cost Matrix using Cosine Similarity
num_a, num_b = len(embeds_a), len(embeds_b)
cost_matrix = np.zeros((num_a, num_b))

for i in range(num_a):
    for j in range(num_b):
        # Enforce that objects must share the same class to be paired
        if boxes_a[i]["class"] != boxes_b[j]["class"]:
            cost_matrix[i][j] = 1.0  # Max distance cost for class mismatches
        else:
            # Cosine similarity converted to distance (1 - similarity)
            sim = cosine_similarity(embeds_a[i], embeds_b[j], dim=0).item()
            cost_matrix[i][j] = 1.0 - sim

# 4. Solve the optimal bipartite match pairing problem (Hungarian Algorithm)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 5. Filter pairings based on a similarity threshold
SIMILARITY_THRESHOLD = 0.75  # Min required confidence to consider it a match

print("--- Pair Matching Results ---")
for a_idx, b_idx in zip(row_ind, col_ind):
    cost = cost_matrix[a_idx][b_idx]
    similarity = 1.0 - cost

    if similarity >= SIMILARITY_THRESHOLD:
        print(
            f"Match Found! ViewA Object #{a_idx} (Class {boxes_a[a_idx]['class']}) "
            f"pairs with ViewB Object #{b_idx} (Confidence: {similarity:.2f})"
        )
    else:
        print(f"Skipped weak match between ViewA #{a_idx} and ViewB #{b_idx}")
