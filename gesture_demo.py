import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

class GestureRecognizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define actions - make sure these match what the model was trained on
        self.actions = ['up', 'down', 'left', 'right', 'nothing']
        self.image_size = (224, 224)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize and load model
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the pre-trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.actions))
        model = model.to(self.device)
        
        # Load saved weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}")
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_hand_region(self, frame):
        """Extract hand region using skin color segmentation"""
        # Convert frame to HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a binary mask
        skin_mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)
        
        # Enhance mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Apply mask to original image
        hand_region = cv2.bitwise_and(frame, frame, mask=skin_mask)
        
        return hand_region, skin_mask
    
    def predict(self, frame):
        """Predict gesture from a frame"""
        # Extract hand region
        hand_region, mask = self.extract_hand_region(frame)
        
        # Convert to RGB (PyTorch models expect RGB)
        hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(hand_region_rgb)
        input_batch = tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        predicted_action = self.actions[prediction.item()]
        confidence_value = confidence.item()
        
        return predicted_action, confidence_value, hand_region, mask
    
    def run_demo(self):
        """Run live demo using webcam"""
        print("Starting gesture recognition demo...")
        print("Press 'q' to quit")
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam!")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Flip for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Make prediction
                action, confidence, hand_region, mask = self.predict(frame)
                
                # Display results
                # Add prediction text to original frame
                cv2.putText(frame, f"Action: {action}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display frames
                cv2.imshow("Gesture Recognition", frame)
                cv2.imshow("Hand Detection", hand_region)
                cv2.imshow("Mask", mask)
                
                # Check for key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error in demo: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            print("Demo ended")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gesture Recognition Demo")
    parser.add_argument("--model", type=str, default="best_gesture_model.pth", 
                        help="Path to the trained model file (.pth)")
    args = parser.parse_args()
    
    # Create recognizer and run demo
    recognizer = GestureRecognizer(args.model)
    recognizer.run_demo()