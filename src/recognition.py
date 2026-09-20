import json
import os

import cv2
import numpy as np

from src.face_detection import detect_faces
from src.embeddings import create_face_recognizer, get_embedding


DATABASE_PATH = "database/embeddings.json"

# Initial cosine similarity threshold.
MATCH_THRESHOLD = 0.363


def load_database():
    if not os.path.exists(DATABASE_PATH):
        return {}

    with open(DATABASE_PATH, "r") as file:
        return json.load(file)


def recognize_face(image):
    faces = detect_faces(image)

    if len(faces) == 0:
        return None, 0.0, "No face detected."

    if len(faces) > 1:
        return None, 0.0, "Multiple faces detected."

    query_embedding = get_embedding(image, faces[0])

    database = load_database()

    if not database:
        return None, 0.0, "Database is empty."

    recognizer = create_face_recognizer()

    best_name = None
    best_score = -1.0

    for name, embeddings in database.items():

        for stored_embedding in embeddings:

            stored_embedding = np.array(
                stored_embedding,
                dtype=np.float32
            ).reshape(1, -1)

            score = recognizer.match(
                query_embedding,
                stored_embedding,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score, "Match found."

    return "Unknown", best_score, "Unknown face."