import os
import cv2
import numpy as np
import sys
import subprocess
import pkg_resources

try:
    import face_recognition
except ImportError as e:
    if "face_recognition_models" in str(e):
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "git+https://github.com/ageitgey/face_recognition_models"])
        import face_recognition
    else:
        raise e

from PIL import Image

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.6):
        """
        Initialize the face recognition system.
        
        Args:
            known_faces_dir (str): Directory containing known face images
            tolerance (float): Tolerance for face recognition (lower is stricter)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create known_faces directory if it doesn't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory: {known_faces_dir}")
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from the known_faces directory."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Extract name from filename (remove extension)
                name = os.path.splitext(filename)[0].replace('_', ' ')
                
                # Load image and get face encoding
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Try to find a face in the image
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    # Use the first face found in the image
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    print(f"Loaded known face: {name}")
                else:
                    print(f"No face found in {filename}")
        
        print(f"Loaded {len(self.known_face_encodings)} known faces")
    
    def add_face(self, image, name):
        """
        Add a new face to the known faces database.
        
        Args:
            image: Image containing a face
            name (str): Name of the person
        
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        # Convert name to filename format
        filename = name.replace(' ', '_') + '.jpg'
        file_path = os.path.join(self.known_faces_dir, filename)
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return False, "No face detected in the image"
        
        if len(face_locations) > 1:
            return False, "Multiple faces detected in the image"
        
        # Get the face encoding
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Save the face image
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(file_path)
        
        # Add to known faces
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        
        return True, f"Face added successfully as {name}"
    
    def recognize_faces(self, image):
        """
        Recognize faces in an image.
        
        Args:
            image: Image to process
        
        Returns:
            tuple: (face_locations, face_names)
        """
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            if True in matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        
        return face_locations, face_names
    
    def draw_faces(self, image, face_locations, face_names):
        """
        Draw rectangles and labels for faces on the image.
        
        Args:
            image: Image to draw on
            face_locations: List of face locations
            face_names: List of face names
        
        Returns:
            image: Image with faces drawn
        """
        # Make a copy to avoid modifying the original
        image_copy = image.copy()
        
        # Draw a rectangle around each face and label with name
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle
            cv2.rectangle(image_copy, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label background
            cv2.rectangle(image_copy, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            
            # Draw name text
            cv2.putText(image_copy, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return image_copy 