# Subway Surfers Gesture Controller

This project provides a gesture-based controller for the game "Subway Surfers" using a webcam and a pre-trained machine learning model. The system tracks hand gestures in real-time and maps them to keyboard actions, enabling hands-free control of the game.

## Features

- **Gesture Control**: Control Subway Surfers using hand gestures, such as:
  - **Up Gesture**: Jump (Up arrow key)
  - **Down Gesture**: Roll (Down arrow key)
  - **Left Gesture**: Move left (Left arrow key)
  - **Right Gesture**: Move right (Right arrow key)
  - **No Gesture**: No action

- **Model Integration**: Uses a pre-trained machine learning model (e.g., ResNet-18) to classify hand gestures.
- **Real-Time Action Execution**: Actions are mapped to keyboard presses using `pyautogui` to simulate key events.
- **Confidence Threshold**: Only gestures with a confidence level above a predefined threshold are executed to avoid false positives.
- **Pause/Resume**: Pause and resume gesture control using the 'p' key.
- **Webcam Interface**: Displays the live webcam feed and feedback about detected actions.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- PyTorch
- `torchvision`
- `pyautogui`
- NumPy

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/metthunder/GestureSurf
   cd GestureSurf
