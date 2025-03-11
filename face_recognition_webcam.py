#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
from face_recognition_utils import FaceRecognitionSystem

def main():
    parser = argparse.ArgumentParser(description='Face Recognition from Webcam')
    parser.add_argument('--tolerance', type=float, default=0.6,
                        help='Tolerance for face recognition (lower is stricter)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--scale', type=float, default=0.25,
                        help='Scale factor for processing (smaller is faster)')
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem(tolerance=args.tolerance)
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(args.camera)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit")
    print("Press 's' to save a screenshot")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
        
        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Recognize faces
        face_locations, face_names = face_system.recognize_faces(rgb_small_frame)
        
        # Scale back face locations
        face_locations = [(int(top / args.scale), int(right / args.scale), 
                          int(bottom / args.scale), int(left / args.scale)) 
                         for top, right, bottom, left in face_locations]
        
        # Draw faces on the frame
        frame = face_system.draw_faces(frame, face_locations, face_names)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Quit on 'q' key
        if key == ord('q'):
            break
        
        # Save screenshot on 's' key
        elif key == ord('s'):
            screenshot_path = f"screenshot_{len(face_names)}_faces.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved to {screenshot_path}")
    
    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 