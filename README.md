# Face Recognition Identification System

An AI/ML-based face recognition system that enrolls individuals, generates face embeddings, identifies known faces using similarity matching, and rejects unknown faces using a configurable similarity threshold.

## Features

- Face detection using OpenCV YuNet
- Face alignment and embedding generation using OpenCV SFace
- Multiple embeddings can be stored for an enrolled person
- Cosine similarity-based face matching
- Unknown-face rejection using a configurable threshold
- Webcam-based enrollment and recognition
- Basic evaluation using known and unknown test images
- False acceptance and rejection analysis

## Project Structure

```text
Face-Recognition-Identification-System/
│
├── app.py
├── requirements.txt
├── README.md
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
├── test_images/
│   ├── known/
│   └── unknown/
│
├── database/
└── results/