import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import sys

class GestureRecognitionPipeline:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.actions = ['up', 'down', 'left', 'right', 'nothing']
        self.image_size = (224, 224)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = self._create_model()
        
        # Load pre-trained model if path is provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def _create_model(self):
        """Create the neural network model"""
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.actions))
        return model.to(self.device)
    
    def extract_hand_region(self, frame):
        """Extract hand region using skin color segmentation"""
        try:
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
        except Exception as e:
            print(f"Error in hand region extraction: {e}")
            return frame, np.zeros_like(frame[:, :, 0])
    
    def preprocess_for_model(self, frame):
        """Preprocess frame for model input"""
        try:
            # Extract hand region
            hand_region, mask = self.extract_hand_region(frame)
            
            # Convert to RGB (PyTorch models expect RGB)
            hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            tensor = self.transform(hand_region_rgb)
            
            return tensor, hand_region, mask
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return dummy tensor and original frame as fallback
            dummy_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]))
            return dummy_tensor, frame, np.zeros_like(frame[:, :, 0])
    
    def prepare_training_dataset(self, data_path):
        """Prepare dataset for training from collected images"""
        X = []  # Images
        y = []  # Labels
        
        print("Processing training data...")
        
        if not os.path.exists(data_path):
            print(f"Error: Data path '{data_path}' does not exist!")
            return torch.tensor([]), torch.tensor([])
        
        for idx, action in enumerate(self.actions):
            action_path = os.path.join(data_path, action)
            
            if not os.path.exists(action_path):
                print(f"Warning: Path for action '{action}' does not exist: {action_path}")
                continue
                
            image_files = [f for f in os.listdir(action_path) if f.endswith('.jpg')]
            total_files = len(image_files)
            
            if total_files == 0:
                print(f"Warning: No images found for action '{action}'")
                continue
                
            print(f"Found {total_files} images for '{action}'")
            processed = 0
            
            for img_file in image_files:
                img_path = os.path.join(action_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                try:
                    # Extract hand region and preprocess
                    hand_region, _ = self.extract_hand_region(img)
                    hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(hand_region_rgb)
                    
                    X.append(tensor)
                    y.append(idx)
                    
                    processed += 1
                    if processed % 20 == 0:
                        print(f"Processed {processed}/{total_files} images for '{action}'")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
        
        if len(X) == 0:
            print("Error: No valid images were processed!")
            return torch.tensor([]), torch.tensor([])
            
        # Convert to PyTorch tensors
        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.long)
        
        print(f"Dataset prepared: {len(X)} images with {len(set(y.tolist()))} classes")
        return X, y
    
    def train_model(self, X, y, batch_size=32, epochs=8, learning_rate=0.001):
        """Train the model on preprocessed data"""
        if len(X) == 0 or len(y) == 0:
            print("Error: Empty dataset, cannot train model")
            return 0.0
            
        try:
            # Split data into training and validation sets
            indices = torch.randperm(len(X))
            train_idx = indices[:int(0.8 * len(X))]
            val_idx = indices[int(0.8 * len(X)):]
            
            train_X, train_y = X[train_idx], y[train_idx]
            val_X, val_y = X[val_idx], y[val_idx]
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
            val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            best_accuracy = 0.0
            
            print("Training model...")
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                train_loss = running_loss / len(train_loader)
                train_acc = 100 * correct / total
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    torch.save(self.model.state_dict(), 'best_gesture_model.pth')
                    print(f"Model saved with accuracy: {val_acc:.2f}%")
            
            print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
            return best_accuracy
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def predict_gesture(self, frame):
        """Predict gesture from a single frame"""
        try:
            # Preprocess frame
            input_tensor, processed_frame, mask = self.preprocess_for_model(frame)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            predicted_action = self.actions[prediction.item()]
            confidence_value = confidence.item()
            
            return predicted_action, confidence_value, processed_frame, mask
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0, frame, np.zeros_like(frame[:, :, 0])
    
    def visualize_prediction(self, frame, action, confidence):
        """Visualize the prediction on the frame"""
        try:
            # Draw prediction text
            cv2.putText(frame, f"Action: {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return frame
        except Exception as e:
            print(f"Error in visualization: {e}")
            return frame
    
    def demo_live_prediction(self):
        """Demo live prediction from webcam"""
        print("Starting live prediction demo...")
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam!")
                return
                
            print("Webcam opened successfully. Press 'q' to exit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame!")
                    break
                
                # Flip for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Predict gesture
                action, confidence, processed, mask = self.predict_gesture(frame)
                
                # Visualize
                result = self.visualize_prediction(frame.copy(), action, confidence)
                
                # Show frames
                cv2.imshow("Original", result)
                cv2.imshow("Hand Detection", processed)
                cv2.imshow("Mask", mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting demo on user command...")
                    break
        
        except Exception as e:
            print(f"Error in demo: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up resources...")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()


def run_training(data_path):
    """Run the training process"""
    try:
        pipeline = GestureRecognitionPipeline()
        X, y = pipeline.prepare_training_dataset(data_path)
        if len(X) > 0:
            pipeline.train_model(X, y)
        else:
            print("No data to train on. Please check your dataset.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def run_demo(model_path="best_gesture_model.pth"):
    """Run the demo process"""
    try:
        pipeline = GestureRecognitionPipeline(model_path)
        pipeline.demo_live_prediction()
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Gesture Recognition Pipeline')
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'demo', 'both'], 
                        help='Mode to run: train, demo, or both')
    parser.add_argument('--data', type=str, default='data', help='Path to training data')
    parser.add_argument('--model', type=str, default='best_gesture_model.pth', help='Path to model file')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train' or args.mode == 'both':
            print("Starting training process...")
            run_training(args.data)
            
        if args.mode == 'demo' or args.mode == 'both':
            print("Starting demo process...")
            run_demo(args.model)
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        
    print("Program completed.")