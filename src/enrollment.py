import json
import os

from src.face_detection import detect_faces
from src.embeddings import get_embedding


DATABASE_PATH = "database/embeddings.json"


def load_database():
    if not os.path.exists(DATABASE_PATH):
        return {}

    with open(DATABASE_PATH, "r") as file:
        return json.load(file)


def save_database(database):
    os.makedirs("database", exist_ok=True)

    with open(DATABASE_PATH, "w") as file:
        json.dump(database, file, indent=4)


def enroll_person(name, image):
    faces = detect_faces(image)

    if len(faces) == 0:
        return False, "No face detected."

    if len(faces) > 1:
        return False, "Multiple faces detected. Please use an image with one person."

    embedding = get_embedding(image, faces[0])

    database = load_database()

    if name not in database:
        database[name] = []

    database[name].append(embedding.flatten().tolist())

    save_database(database)

    return True, f"{name} enrolled successfully."