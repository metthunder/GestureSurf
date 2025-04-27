import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import pyautogui
import time
import os

class SubwaySurfersGestureController:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define actions - make sure these match what the model was trained on
        self.actions = ['up', 'down', 'left', 'right', 'nothing']
        self.image_size = (224, 224)
        
        # Keyboard mapping - map gestures to keys
        self.key_mapping = {
            'up': 'up',     # Jump
            'down': 'down', # Roll
            'left': 'left', # Move left
            'right': 'right', # Move right
            'nothing': None  # No action
        }
        
        # Control variables
        self.last_action = 'nothing'
        self.action_cooldown = 0.3  # seconds between actions to prevent spamming
        self.last_action_time = time.time()
        self.confidence_threshold = 0.6  # minimum confidence to trigger an action
        
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
        
    def execute_action(self, action, confidence):
        """Execute the corresponding keyboard command based on gesture"""
        current_time = time.time()
        
        # Check if action meets confidence threshold and cooldown period
        if (confidence >= self.confidence_threshold and 
            current_time - self.last_action_time >= self.action_cooldown):
            
            # If this is a new action or a repeated directional action
            if action != self.last_action or action in ['left', 'right', 'up', 'down']:
                key = self.key_mapping.get(action)
                
                if key is not None:
                    # Release previous key if it was a direction
                    if self.last_action in ['left', 'right', 'up', 'down']:
                        prev_key = self.key_mapping.get(self.last_action)
                        if prev_key:
                            pyautogui.keyUp(prev_key)
                    
                    # For directional controls, press and hold
                    if action in ['left', 'right']:
                        pyautogui.keyDown(key)
                    # For jump/roll, just press briefly
                    elif action in ['up', 'down']:
                        pyautogui.press(key)
                        
                    print(f"Executing: {action} (confidence: {confidence:.2f})")
                    self.last_action_time = current_time
                    self.last_action = action
                    return True
        
        # Release keys when "nothing" is detected with high confidence
        if action == 'nothing' and confidence >= self.confidence_threshold:
            if self.last_action in ['left', 'right', 'up', 'down']:
                prev_key = self.key_mapping.get(self.last_action)
                if prev_key:
                    pyautogui.keyUp(prev_key)
                self.last_action = 'nothing'
                
        return False
    
    def run_controller(self):
        """Run the gesture controller for Subway Surfers"""
        print("Starting Subway Surfers Gesture Controller")
        print("-------------------------------------------")
        print("1. Open Subway Surfers game")
        print("2. Position the game window so it's visible while the webcam can see your gestures")
        print("3. Use hand gestures to control the game:")
        print("   - Up gesture: Jump")
        print("   - Down gesture: Roll/Slide")
        print("   - Left gesture: Move left")
        print("   - Right gesture: Move right")
        print("   - No gesture: No action")
        print("\nPress 'q' to quit, 'p' to pause/resume control")
        print("-------------------------------------------")
        print("Starting in 5 seconds... Position your game window!")
        time.sleep(5)
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam!")
                return
            
            paused = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break
                
                # Flip for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Make prediction
                action, confidence, hand_region, mask = self.predict(frame)
                
                # Execute action if not paused
                action_executed = False
                if not paused:
                    action_executed = self.execute_action(action, confidence)
                
                # Display status
                status_color = (0, 255, 0) if not paused else (0, 0, 255)
                status_text = "ACTIVE" if not paused else "PAUSED"
                
                # Display results
                cv2.putText(frame, f"Action: {action}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f"Status: {status_text}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Add visual indicator for executed actions
                if action_executed:
                    cv2.putText(frame, "ACTION!", (frame.shape[1]-150, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Display frames
                cv2.imshow("Subway Surfers Controller", frame)
                cv2.imshow("Hand Detection", hand_region)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"Controller {'paused' if paused else 'resumed'}")
                    # Release all keys when pausing
                    if paused:
                        for action in ['left', 'right', 'up', 'down']:
                            key = self.key_mapping.get(action)
                            if key:
                                pyautogui.keyUp(key)
                
        except Exception as e:
            print(f"Error in controller: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release all keys
            for action in ['left', 'right', 'up', 'down']:
                key = self.key_mapping.get(action)
                if key:
                    pyautogui.keyUp(key)
                    
            # Release resources
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            print("Controller ended")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Subway Surfers Gesture Controller")
    parser.add_argument("--model", type=str, default="best_gesture_model.pth", 
                        help="Path to the trained model file (.pth)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence threshold for actions (0.0-1.0)")
    parser.add_argument("--cooldown", type=float, default=0.3,
                        help="Cooldown time between actions in seconds")
    
    args = parser.parse_args()
    
    # Create controller and run
    controller = SubwaySurfersGestureController(args.model)
    controller.confidence_threshold = args.threshold
    controller.action_cooldown = args.cooldown
    controller.run_controller()