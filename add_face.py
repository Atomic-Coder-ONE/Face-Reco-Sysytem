#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import face_recognition
from face_recognition_utils import FaceRecognitionSystem

def add_face_from_webcam(face_system, name):
    """Add a face from webcam capture."""
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    print("Press 'c' to capture your face")
    print("Press 'q' to quit without capturing")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw rectangles around faces
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display instructions on the frame
        cv2.putText(frame, f"Name: {name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the number of faces detected
        cv2.putText(frame, f"Faces detected: {len(face_locations)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Capture Face', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Capture on 'c' key
        if key == ord('c'):
            if len(face_locations) == 1:
                success, message = face_system.add_face(rgb_frame, name)
                video_capture.release()
                cv2.destroyAllWindows()
                return success, message
            elif len(face_locations) == 0:
                print("No face detected. Please position your face in the frame.")
            else:
                print("Multiple faces detected. Please ensure only one face is in the frame.")
        
        # Quit on 'q' key
        elif key == ord('q'):
            break
    
    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    return False, "Face capture cancelled"

def add_face_from_image(face_system, name, image_path):
    """Add a face from an image file."""
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Add the face
        success, message = face_system.add_face(image, name)
        return success, message
    
    except Exception as e:
        return False, f"Error processing image: {e}"

def main():
    parser = argparse.ArgumentParser(description='Add Face to Recognition Database')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the person')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file (optional, uses webcam if not provided)')
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem()
    
    if args.image:
        # Add face from image file
        success, message = add_face_from_image(face_system, args.name, args.image)
    else:
        # Add face from webcam
        success, message = add_face_from_webcam(face_system, args.name)
    
    # Print result
    if success:
        print(f"Success: {message}")
    else:
        print(f"Failed: {message}")

if __name__ == "__main__":
    main() 