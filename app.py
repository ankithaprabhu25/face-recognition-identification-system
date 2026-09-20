import cv2

from src.enrollment import enroll_person
from src.recognition import recognize_face


def capture_from_webcam():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Webcam opened.")
    print("Press SPACE to capture your face.")
    print("Press ESC to cancel.")

    captured_image = None

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read webcam.")
            break

        cv2.imshow("Face Recognition - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_image = frame.copy()
            break

        if key == 27:  # ESC
            break

    camera.release()
    cv2.destroyAllWindows()

    return captured_image


def enroll_from_webcam():
    name = input("Enter person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    image = capture_from_webcam()

    if image is None:
        return

    success, message = enroll_person(name, image)
    print(message)


def recognize_from_webcam():
    image = capture_from_webcam()

    if image is None:
        return

    name, score, message = recognize_face(image)

    print("\n===== Recognition Result =====")
    print(f"Name: {name}")
    print(f"Similarity score: {score:.4f}")
    print(f"Status: {message}")


def main():
    while True:
        print("\n===== Face Recognition Identification System =====")
        print("1. Enroll person using webcam")
        print("2. Recognize face using webcam")
        print("3. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            enroll_from_webcam()

        elif choice == "2":
            recognize_from_webcam()

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()