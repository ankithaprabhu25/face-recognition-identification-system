import cv2


MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


def create_face_detector():
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 640),
        0.5,
        0.3,
        5000
    )

    return detector


def detect_faces(image):
    detector = create_face_detector()

    original_height, original_width = image.shape[:2]

    # Resize while preserving aspect ratio
    max_dimension = 960

    scale = min(
        1.0,
        max_dimension / max(original_width, original_height)
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(
        image,
        (new_width, new_height)
    )

    # Tell YuNet the actual image size
    detector.setInputSize(
        (new_width, new_height)
    )

    _, faces = detector.detect(resized_image)

    if faces is None:
        return []

    faces = faces.copy()

    # Convert bounding box and landmark coordinates
    # back to original image coordinates.
    faces[:, :14] /= scale

    return faces