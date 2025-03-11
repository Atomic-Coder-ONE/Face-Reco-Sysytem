import cv2
import numpy as np
import os
import gdown
from pathlib import Path

class FaceAnalyzer:
    def __init__(self):
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Paths for the models
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        self.age_model_path = model_dir / "age_net.caffemodel"
        self.age_proto_path = model_dir / "age_deploy.prototxt"
        self.gender_model_path = model_dir / "gender_net.caffemodel"
        self.gender_proto_path = model_dir / "gender_deploy.prototxt"
        
        # Download models if they don't exist
        self.download_models()
        
        # Load models
        self.age_net = cv2.dnn.readNet(str(self.age_model_path), str(self.age_proto_path))
        self.gender_net = cv2.dnn.readNet(str(self.gender_model_path), str(self.gender_proto_path))
        
        # Model parameters
        self.age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        self.gender_list = ['Male', 'Female']
        
    def download_models(self):
        """Download pre-trained models if they don't exist."""
        model_files = {
            self.age_model_path: "https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW",
            self.age_proto_path: "https://drive.google.com/uc?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl",
            self.gender_model_path: "https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ",
            self.gender_proto_path: "https://drive.google.com/uc?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP"
        }
        
        for file_path, url in model_files.items():
            if not file_path.exists():
                print(f"Downloading {file_path.name}...")
                gdown.download(url, str(file_path), quiet=False)
    
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return faces
    
    def predict_age_gender(self, face_img):
        """Predict age and gender using deep learning models."""
        # Preprocess the face image
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
        
        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        gender_conf = float(gender_preds[0].max() * 100)
        
        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        age_conf = float(age_preds[0].max() * 100)
        
        return gender, gender_conf, age, age_conf
    
    def draw_results(self, frame, faces):
        """Draw detection results on the frame."""
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract and analyze face
                face_img = frame[y:y+h, x:x+w]
                
                # Skip if face region is invalid
                if face_img.size == 0:
                    continue
                
                # Get predictions
                gender, gender_conf, age, age_conf = self.predict_age_gender(face_img)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Prepare label text
                label = f"{gender} ({gender_conf:.1f}%)"
                label += f"\nAge: {age} ({age_conf:.1f}%)"
                
                # Draw background for text
                cv2.rectangle(frame, (x, y-60), (x+w, y), (0, 255, 0), -1)
                
                # Draw text (split into lines)
                y_pos = y - 40
                for line in label.split('\n'):
                    cv2.putText(frame, line, (x+5, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 255, 255), 2)
                    y_pos += 20
        
        return frame

def main():
    # Initialize the face analyzer
    print("Initializing face analyzer (this may take a moment to download models)...")
    analyzer = FaceAnalyzer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face Analysis Started")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect faces
        faces = analyzer.detect_faces(frame)
        
        # Draw results
        frame = analyzer.draw_results(frame, faces)
        
        # Display number of faces detected
        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Analysis', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{len(faces)}_faces.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 