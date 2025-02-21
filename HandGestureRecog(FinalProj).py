import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('gesture_model.keras')

# Map gesture IDs to labels
gesture_labels = {
    0: 'Open Palm',
    1: 'Fist',
    2: 'Okay',
    3: 'Peace',
    4: "Call me",
    5: "L",
    6: "None"
    # Add more gestures as needed
}

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils

def gesture_recognition():
    webcam = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while True:
            ret, frame = webcam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawings.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks for gesture recognition
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y])
                    landmark_array = np.array(landmark_list, dtype='float32').reshape(1, -1)

                    # Predict gesture
                    prediction = model.predict(landmark_array, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Analyze prediction confidence
                    if confidence > 0.85:  # Use confidence threshold
                        if confidence > 0.95 or np.std(prediction) > 0.05:  # Variance check
                            gesture_name = gesture_labels.get(predicted_class, "Unknown")
                        else:
                            gesture_name = "None"
                    else:
                        gesture_name = "None"

                    display_text = f"Gesture: {gesture_name} ({confidence:.2f})"
                    # Display gesture name and confidence
                    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                # Display "None" when no hands are detected
                cv2.putText(frame, "Gesture: None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    gesture_recognition()
