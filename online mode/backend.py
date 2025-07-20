from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    landmarks: list[float]


# Load your trained model
try:
    model_data = joblib.load('model.pkl')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    # Create fallback model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    model = RandomForestClassifier(n_estimators=10)
    X = np.random.rand(10, 63)
    y = np.random.randint(0, 7, 10)
    model.fit(X, y)
    model_data = {
        'model': model,
        'label_encoder': LabelEncoder().fit(
            ['Stop', 'Move Up', 'Move Left', 'Move Right', 'Move Down', 'Zoom Out', 'Zoom In'])
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        landmarks = np.array(request.landmarks, dtype=np.float32).reshape(1, -1)

        if landmarks.shape[1] != 63:
            raise ValueError("Expected 63 landmarks (21 points with x,y,z)")

        # Get prediction
        prediction = model_data['model'].predict(landmarks)[0]
        gesture = model_data['label_encoder'].inverse_transform([prediction])[0]

        # Get confidence
        confidence = 1.0
        if hasattr(model_data['model'], 'predict_proba'):
            confidence = model_data['model'].predict_proba(landmarks).max()

        return {
            "gesture": gesture,
            "confidence": float(confidence),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)