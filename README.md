# Face Recognition Identification System

A real-time AI/ML-based face recognition web application that detects faces through a webcam, generates face embeddings using OpenCV SFace, compares them with enrolled faces using cosine similarity, and rejects unknown faces using a similarity threshold.

## Features

- Real-time webcam face capture
- Face detection using OpenCV YuNet
- Face alignment and face embeddings using OpenCV SFace
- Cosine similarity-based face matching
- Unknown-face rejection using a configurable threshold
- Web-based interface using Flask
- Simple enrollment of new people through the webcam
- Captured face preview
- Displays identity and similarity score
- Supports multiple embeddings for an enrolled person
- Basic evaluation using known and unknown test images

## System Flow

Webcam
   ↓
Face Detection
   ↓
YuNet
   ↓
Face Alignment
   ↓
SFace
   ↓
Face Embedding
   ↓
Cosine Similarity
   ↓
Compare with Enrolled Faces
   ↓
Similarity ≥ 0.363 ?
   ↓
Known Person / Unknown

## Technologies Used

- Python
- Flask
- OpenCV
- YuNet
- SFace
- NumPy
- HTML / CSS / JavaScript

## How It Works

### 1. Face Detection

The webcam captures an image and OpenCV YuNet detects the face. The system also checks whether no face, one face, or multiple faces are detected.

The current system requires exactly one face for enrollment and recognition.

### 2. Face Alignment and Embedding

The detected face is aligned using OpenCV SFace. SFace then converts the face into a numerical embedding that represents the facial features.

### 3. Face Matching

The new face embedding is compared with the embeddings of enrolled people using cosine similarity.

The system selects the highest similarity score as the best match.

### 4. Unknown Face Rejection

The system uses a similarity threshold of 0.363.

Similarity ≥ 0.363  →  Recognized person
Similarity < 0.363  →  Unknown

For example:

Ankitha (0.8338)

means the best matching enrolled identity is Ankitha with a similarity score of 0.8338.

Unknown (0.2614)

means the best similarity score is below the threshold, so the person is rejected as unknown.

## Enrollment

To enroll a person:

1. Start the webcam.
2. Position the person's face inside the camera frame.
3. Capture the face.
4. Enter the person's name.
5. Click "Enroll Captured Face".
6. YuNet detects the face.
7. SFace generates the face embedding.
8. The embedding is stored for future identification.

The enrolled embeddings are stored locally in:

database/embeddings.json

Multiple embeddings can be stored for the same person.

The biometric database is not included in the public GitHub repository.

## Screenshots

### 1. Person Enrollment

A person can be enrolled by entering their name and capturing their face through the webcam.

![Person Enrollment](screenshots/enrollment.png)

### 2. Known Person Recognition

After enrollment, the system recognizes the person and displays the identity with the similarity score.

![Known Person Recognition](screenshots/recognition.png)

### 3. Unknown Person Rejection

When a person who is not enrolled is detected, the system displays "Unknown" because the similarity score is below the threshold.

![Unknown Person Rejection](screenshots/unknown.png)

## Project Structure

Face-Recognition-Identification-System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── src/
│   ├── __init__.py
│   ├── face_detection.py
│   ├── embeddings.py
│   ├── enrollment.py
│   ├── recognition.py
│   └── evaluation.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── database/
│   └── embeddings.json
│
├── test_images/
│   ├── known/
│   └── unknown/
│
├── results/
│
└── screenshots/
    ├── enrollment.png
    ├── recognition.png
    └── unknown.png

## Installation

### 1. Clone the repository

git clone https://github.com/ankithaprabhu25/face-recognition-identification-system.git

cd face-recognition-identification-system

### 2. Create a virtual environment

python -m venv venv

### 3. Activate the virtual environment

Windows:

venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

## Run the Application

python app.py

Open the following address in your browser:

http://127.0.0.1:5000

Allow webcam access when requested.

## Testing

The system can be tested using known and unknown faces.

### Known Person

1. Enroll a person.
2. Stand in front of the webcam again.
3. Capture the face.
4. Click "Recognize Captured Face".
5. The system should display the enrolled person's name and similarity score.

### Unknown Person

1. Enroll one person.
2. Do not enroll the second person.
3. Let the second person stand in front of the webcam.
4. Capture the face.
5. Click "Recognize Captured Face".
6. The system should display "Unknown" when the similarity score is below the threshold.

### Evaluation Dataset

1. Enrollment Process
   <img width="1206" height="954" alt="Screenshot 2026-09-21 151631" src="https://github.com/user-attachments/assets/17d8c54d-d276-403e-85f9-15c588070838" />

2. Identifying Enrolled Person
   <img width="1094" height="882" alt="Screenshot 2026-09-21 151712" src="https://github.com/user-attachments/assets/1360f178-e542-46ae-b3bd-2ebb068183cf" />

3. Detection of Unknown Person
   <img width="498" height="759" alt="Screenshot 2026-09-21 151820" src="https://github.com/user-attachments/assets/bb4c9f5f-59a3-411a-8914-68f4029a6123" />

### Evaluation Results

| Metric | Result |
|---|---:|
| Known-face identification accuracy | 100.00% |
| Unknown-face rejection rate | 100.00% |
| False acceptance rate | 0.00% |

These results are based on a small test dataset and should not be considered a general real-world accuracy guarantee.

### Different Conditions

The system can also be tested under:

- Different lighting conditions
- Different face positions
- Slight changes in facial appearance
- Multiple faces in the frame
- Blurry or partially visible faces
- Different distances from the camera

## Failure Cases

The recognition result can be affected by:

- Poor lighting
- Blurry images
- Partial face visibility
- Extreme face angles
- Very small faces
- Multiple faces in the frame
- Significant changes in appearance
- Low-quality webcam images

The system also handles:

- No face detected
- Multiple faces detected
- Empty database
- Unknown face
- Camera access failure

## Possible Improvements

- Use multiple and more diverse images during enrollment
- Add face quality checking
- Improve handling of different lighting conditions
- Add liveness detection
- Add anti-spoofing
- Add face tracking for smoother recognition
- Calibrate the similarity threshold using a larger test dataset
- Support multiple-face recognition
- Add encrypted biometric storage
- Add formal evaluation metrics such as precision, recall, F1-score, FAR and FRR

## Conclusion

This project demonstrates a complete real-time face recognition pipeline:

Face Detection → Face Alignment → Face Embedding → Similarity Matching → Unknown Rejection

The system provides a simple Flask web interface for enrollment and real-time identification using a webcam, OpenCV YuNet, and SFace.
