import cv2


MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"


def create_face_recognizer():
    recognizer = cv2.FaceRecognizerSF.create(
        MODEL_PATH,
        ""
    )

    return recognizer


def get_embedding(image, face):
    recognizer = create_face_recognizer()

    aligned_face = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned_face)

    return embedding