import cv2
import numpy as np
import os
import time

def collect_training_data(actions, data_path='data'):
    """
    Collect training data for each action with improved frame capture
    """
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create directories for data
    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
    
    # Collect images for each action
    for action in actions:
        print(f"Collecting data for {action}")
        print("Press 'q' to stop collection for this gesture or 'x' to cancel all")
        
        # Wait for readiness
        input(f"Press ENTER to start collecting data for '{action}'...")
        
        # Countdown before starting
        for countdown in range(3, 0, -1):
            print(f"{countdown}...")
            time.sleep(1)
        print("GO! Show your gesture now")
        
        num_samples = 0
        frame_count = 0  # Counter for all frames (to capture every 3rd)
        
        while num_samples < 100:  # Collect 200 images per action
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for more intuitive interactions
            frame = cv2.flip(frame, 1)
            
            # Copy frame for display
            display_frame = frame.copy()
            
            # Display instructions and progress
            cv2.putText(display_frame, f"Collecting: {action} ({num_samples}/200)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Collecting Gesture Data', display_frame)
            
            # Save frame (every 3rd frame to avoid too similar images)
            frame_count += 1
            if frame_count % 3 == 0:
                img_path = os.path.join(data_path, action, f"{action}_{num_samples}.jpg")
                cv2.imwrite(img_path, frame)
                num_samples += 1
                print(f"Saved image {num_samples}/200 for {action}")
            
            # Check for key press (wait 10ms)
            key = cv2.waitKey(10)
            if key == ord('q'):
                print(f"Stopped collection for {action} with {num_samples} samples")
                break
            elif key == ord('x'):
                print("Cancelling all data collection")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        print(f"Completed collection for {action} with {num_samples} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")

# Test the function directly
if __name__ == "__main__":
    actions = ['up', 'down', 'left', 'right', 'nothing']
    collect_training_data(actions)