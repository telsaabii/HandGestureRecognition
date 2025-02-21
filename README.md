
- Goal: Implement a system to detect, classify, and track hand gestures in real time using a webcam.  
- Technologies:  
  - [OpenCV](https://opencv.org/) (computer vision)  
  - [MediaPipe](https://developers.google.com/mediapipe) (hand landmark detection)  
  - [TensorFlow / Keras](https://www.tensorflow.org/) (model training and inference)  
  - [PyQt5 + PyQtGraph](https://www.pyqtgraph.org/) (visualization in `handtracking.py`)
   
Installation and Dependencies
1. Python version: 3.7+ recommended.
2. pip install opencv-python mediapipe tensorflow scikit-learn pyqt5 pyqtgraph
3. You may also need:
   - tkinter (often included by default on many systems)
   - numpy
   - csv (standard library in Python 3)


Usage

*Make sure your webcam is connected and accessible.  

1. Data Collection
   - Run `datacollection.py` to capture hand landmarks and label them.  
   - Press `c` to start/stop data collection.  
   - Use number keys (0–9) as gesture labels while capturing.  
   - Press `q` to quit and save to `gestures.csv`.  

2. Model Training
   - Run `modeltraining.py` to train a gesture-classification model.  
   - This script reads `gestures.csv`, splits it into training/validation/test sets, and trains a TensorFlow model.  
   - At the end, the script saves the Keras model as `gesture_model.h5` and `gesture_model.keras`. It also optionally creates a `gesture_model.tflite` file.  

3. Run Gesture Recognition (2D Demo) 
   - Run `handgesturerecog.py` to perform real-time gesture recognition using your webcam.  
   - The script loads `gesture_model.keras` and outputs predicted gestures on the video frame.  
   - Press `q` to quit.  

4. Run Hand Tracking (3D Visualization)
   - Run `handtracking.py` to visualize hand landmarks in a 3D space (using PyQt and OpenGL).  
   - A separate thread captures webcam data, while a PyQt window displays 3D points and lines representing hand landmarks.  
   - Press `q` in the webcam display or close the PyQt window to quit.  

ENJOY
