import cv2
import mediapipe as mp
import csv
import time
from tkinter import Tk

# Get screen dimensions
root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils

# Function to capture hand data and save it into a CSV file
def capture_hand_gestures_to_csv():
    capturing = False
    current_label = None

    with open('gestures.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Initialize webcam
        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            while True:
                ret, frame = webcam.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw landmarks on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display instructions
                cv2.putText(frame, "Press 'c' to start/stop capturing. Press number keys to label gestures.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if capturing:
                    cv2.putText(frame, f"Capturing... Label: {current_label}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Hand Gesture Capture", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):  # Quit
                    break
                elif key == ord('c'):  # Toggle capture mode
                    capturing = not capturing
                    current_label = None
                elif key in [ord(str(i)) for i in range(10)] and capturing:  # Set gesture label
                    current_label = int(chr(key))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmark_list = []
                            for lm in hand_landmarks.landmark:
                                landmark_list.extend([lm.x, lm.y])
                            csv_writer.writerow([current_label] + landmark_list)
                            print(f"Saved data for label {current_label}")

                # Delay to prevent high CPU usage
                time.sleep(0.01)

        webcam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_hand_gestures_to_csv()
