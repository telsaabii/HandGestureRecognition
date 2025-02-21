#integrating with 3d environment
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import threading
from threading import Lock
import copy
import time
from tkinter import Tk
import tensorflow as tf
# Load the trained model
model = tf.keras.models.load_model('gesture_model.keras')

# Map gesture IDs to labels
gesture_labels = {
    1: 'Fist',
    # Add more gestures as needed
}
#get screen dimensions
root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils

# Global variables for threading
hand_data = None
hand_data_lock = Lock()
terminate = False

# Function to capture hand data
def capture_hand_data():
    global hand_data
    global terminate
    global predicted_class
    #capturing webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width//2)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height//2)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while not terminate:
            ret, frame = webcam.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame,1)
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(image)

            # Update hand_data
            with hand_data_lock:
                if results.multi_hand_landmarks:
                    hand_data = copy.deepcopy(results.multi_hand_landmarks)
                else:
                    hand_data = None

            # Optional: Display the image with OpenCV
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y])
                    landmark_array = np.array(landmark_list, dtype='float32').reshape(1, -1)

                    # Predict gesture
                    prediction = model.predict(landmark_array, verbose=0)
                    predicted_class = np.argmax(prediction)
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                terminate = True
                break

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

    webcam.release()
    cv2.destroyAllWindows()


# PyQtGraph visualization
class HandVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle('Real-time Hand Tracking')

        # Get screen dimensions
        screen_geometry = QDesktopWidget().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Set the window size to match the screen
        self.setGeometry(0, 0, screen_width, screen_height)

        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtWidgets.QVBoxLayout()
        self.widget.setLayout(self.layout)

        # Set up the OpenGL view
        self.gl_widget = gl.GLViewWidget()
        self.layout.addWidget(self.gl_widget)

        # Create scatter plots for hands
        self.hand_scatter_plots = []

        # Set up the coordinate grid
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self.gl_widget.addItem(grid)

        # Timer for updating the plot
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)  # Update every 30 ms (~33 FPS)


        self.previous_z = None
        self.flash_counter = 0

    def update_plot(self):
        global hand_data
        with hand_data_lock:
            data = copy.deepcopy(hand_data)

        # Clear existing scatter plots
        for scatter in self.hand_scatter_plots:
            self.gl_widget.removeItem(scatter)
        self.hand_scatter_plots = []

        if data is not None:
            for hand_landmarks in data:
                x_vals = np.array([lm.x for lm in hand_landmarks.landmark]) - 0.5
                y_vals = np.array([lm.y for lm in hand_landmarks.landmark]) - 0.5
                z_vals = np.array([lm.z for lm in hand_landmarks.landmark])

                x_vals *= -1
                y_vals *= -1  # Flip orientation to mirror horizontally

                # Scale the hand to make it appear larger
                scale_factor = 2.0
                x_vals *= scale_factor
                y_vals *= scale_factor
                z_vals *= scale_factor

                # Create scatter plot item
                pos = np.vstack([x_vals, z_vals, y_vals]).transpose()

                # Plot the hand landmarks
                scatter = gl.GLScatterPlotItem(pos=pos, color=(0, 1, 0, 1), size=10)
                self.gl_widget.addItem(scatter)
                self.hand_scatter_plots.append(scatter)

                # Draw the connections
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    x = np.array([x_vals[start_idx], x_vals[end_idx]])
                    y = np.array([y_vals[start_idx], y_vals[end_idx]])
                    z = np.array([z_vals[start_idx], z_vals[end_idx]])
                    pos_line = np.vstack([x, z, y]).transpose()
                    line = gl.GLLinePlotItem(pos=pos_line, color=(1, 0, 0, 1), width=2, antialias=True)
                    self.gl_widget.addItem(line)
                    self.hand_scatter_plots.append(line)

                avg_z = np.mean(z_vals)
                print(predicted_class)
                if predicted_class == 1 and self.previous_z is not None:
                    z_delta = avg_z - self.previous_z
                    if z_delta < -0.2:
                        self.flash_counter = 10
                self.previous_z = avg_z 
        if self.flash_counter > 0:
            self.gl_widget.opts['bgcolor'] = (1, 0, 0, 1)  # Red flash
            self.flash_counter -= 1
        else:
            self.gl_widget.opts['bgcolor'] = (0, 0, 0, 1)  # Default background

    def closeEvent(self, event):
        global terminate
        terminate = True
        event.accept()


# Main application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    visualizer = HandVisualizer()

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_hand_data)
    capture_thread.start()

    visualizer.show()
    sys.exit(app.exec_())

    # Wait for the capture thread to finish
    capture_thread.join()