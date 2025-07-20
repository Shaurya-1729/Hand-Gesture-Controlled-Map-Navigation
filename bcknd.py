from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
import mediapipe as mp
import joblib
from pydantic import BaseModel
import os
import logging


logging.basicConfig(level=logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '2'  # Suppress Google logging (MediaPipe)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load your trained model
try:
    model_data = joblib.load('/Users/shauryaawasthi/Hand_gest_project/Final working model/offline_map_project/model.pkl')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    # Create fallback model if loading fails
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    model = RandomForestClassifier(n_estimators=10)
    X = np.random.rand(10, 63)  # 21 landmarks * 3 (x,y,z)
    y = np.random.randint(0, 7, 10)
    model.fit(X, y)
    model_data = {
        'model': model,
        'label_encoder': LabelEncoder().fit(
            ['Stop', 'Move Up', 'Move Left', 'Move Right', 'Move Down', 'Zoom Out', 'Zoom In'])
    }


class FrameRequest(BaseModel):
    image: str  # Base64 encoded image


@app.post("/process_frame")
async def process_frame(request: FrameRequest):
    try:
        # Decode the image
        image_data = base64.b64decode(request.image.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image")

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Extract landmarks in the format your model expects (63 values: 21 points * 3 coordinates)
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Convert to numpy array and reshape for your model
            landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)

            # Get prediction from your model
            prediction = model_data['model'].predict(landmarks_array)[0]
            gesture = model_data['label_encoder'].inverse_transform([prediction])[0]

            # Get confidence
            confidence = 1.0
            if hasattr(model_data['model'], 'predict_proba'):
                confidence = model_data['model'].predict_proba(landmarks_array).max()

            return {
                "gesture": gesture,
                "confidence": float(confidence),
                "status": "success",
                "hand_detected": True
            }

        # Return when no hand is detected
        return {
            "status": "success",
            "hand_detected": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)

    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=8000)