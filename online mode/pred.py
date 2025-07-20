import cv2
import mediapipe as mp
import joblib
import numpy as np


def load_model(model_path='/Users/shauryaawasthi/Hand_gest_project/Initial training/model.pkl'):
    try:
        data = joblib.load(model_path)
        return data['model'], data['label_encoder']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None


def main():
    model, le = load_model()
    if model is None or le is None:
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Mirror the display (optional, doesn't affect prediction)
            frame = cv2.flip(frame, 1)

            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    # Predict
                    try:
                        pred = model.predict([landmarks])[0]
                        gesture = le.inverse_transform([pred])[0]

                        # Swap left/right predictions only
                        if gesture == 'Move Left':
                            gesture = 'Move Right'
                        elif gesture == 'Move Right':
                            gesture = 'Move Left'

                        cv2.putText(frame, gesture, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Prediction error: {e}")

            cv2.imshow('Hand Gesture', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()