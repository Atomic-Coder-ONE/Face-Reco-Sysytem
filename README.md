# Face Recognition System

A simple face recognition system that can detect and recognize faces in images and video streams.

## Features

- Face detection in images and video
- Face recognition against known faces
- Real-time video processing
- Simple user interface for adding new faces to the database

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a directory for known faces:
   ```
   mkdir -p known_faces
   ```

3. Add known face images to the `known_faces` directory with the person's name as the filename (e.g., `john_doe.jpg`).

## Usage

### Face Recognition from Webcam

```
python face_recognition_webcam.py
```

### Face Recognition from Image

```
python face_recognition_image.py --image path/to/image.jpg
```

### Add New Face to Database

```
python add_face.py --name "Person Name"
```

## Requirements

- Python 3.6+
- OpenCV
- face_recognition library (which uses dlib)
- Webcam (for real-time recognition) 