#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
import face_recognition
from face_recognition_utils import FaceRecognitionSystem

def main():
    parser = argparse.ArgumentParser(description='Face Recognition from Image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--tolerance', type=float, default=0.6,
                        help='Tolerance for face recognition (lower is stricter)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output image (optional)')
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem(tolerance=args.tolerance)
    
    # Load the image
    try:
        image = face_recognition.load_image_file(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Convert from RGB to BGR (for OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Recognize faces
    face_locations, face_names = face_system.recognize_faces(image)
    
    # Draw faces on the image
    result_image = face_system.draw_faces(image_bgr, face_locations, face_names)
    
    # Print results
    if len(face_locations) == 0:
        print("No faces found in the image.")
    else:
        print(f"Found {len(face_locations)} faces:")
        for name in face_names:
            print(f"- {name}")
    
    # Save output image if specified
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Output image saved to {args.output}")
    
    # Display the result
    cv2.imshow('Face Recognition Result', result_image)
    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 