import os

import cv2

from src.recognition import recognize_face


KNOWN_FOLDER = "test_images/known"
UNKNOWN_FOLDER = "test_images/unknown"


def evaluate_known_faces():
    total = 0
    correct = 0
    rejected = 0
    no_face = 0

    for filename in os.listdir(KNOWN_FOLDER):

        file_path = os.path.join(KNOWN_FOLDER, filename)

        image = cv2.imread(file_path)

        if image is None:
            continue

        total += 1

        name, score, status = recognize_face(image)

        if name == "Ankitha":
            correct += 1
        elif name == "Unknown":
            rejected += 1
        else:
            no_face += 1

        print(
            f"[KNOWN] {filename} -> "
            f"{name} | score={score:.4f}"
        )

    return total, correct, rejected, no_face


def evaluate_unknown_faces():
    total = 0
    rejected = 0
    false_accepts = 0
    no_face = 0

    for filename in os.listdir(UNKNOWN_FOLDER):

        file_path = os.path.join(UNKNOWN_FOLDER, filename)

        image = cv2.imread(file_path)

        if image is None:
            continue

        total += 1

        name, score, status = recognize_face(image)

        if name == "Unknown":
            rejected += 1
        elif name is None:
            no_face += 1
        else:
            false_accepts += 1

        print(
            f"[UNKNOWN] {filename} -> "
            f"{name} | score={score:.4f}"
        )

    return total, rejected, false_accepts, no_face


def run_evaluation():

    (
        known_total,
        known_correct,
        known_rejected,
        known_no_face
    ) = evaluate_known_faces()

    (
        unknown_total,
        unknown_rejected,
        unknown_false_accepts,
        unknown_no_face
    ) = evaluate_unknown_faces()

    known_accuracy = (
        known_correct / known_total * 100
        if known_total > 0 else 0
    )

    unknown_rejection_rate = (
        unknown_rejected / unknown_total * 100
        if unknown_total > 0 else 0
    )

    false_acceptance_rate = (
        unknown_false_accepts / unknown_total * 100
        if unknown_total > 0 else 0
    )

    print("\n===== Evaluation Results =====")

    print(
        f"Known-face identification accuracy: "
        f"{known_accuracy:.2f}%"
    )

    print(
        f"Unknown-face rejection rate: "
        f"{unknown_rejection_rate:.2f}%"
    )

    print(
        f"False acceptance rate: "
        f"{false_acceptance_rate:.2f}%"
    )

    print(
        f"Known images rejected: "
        f"{known_rejected}"
    )

    print(
        f"Known images with no face: "
        f"{known_no_face}"
    )

    print(
        f"Unknown images with no face: "
        f"{unknown_no_face}"
    )


if __name__ == "__main__":
    run_evaluation()