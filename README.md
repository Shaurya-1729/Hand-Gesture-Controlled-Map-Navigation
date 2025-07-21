# Hand-Gesture-Controlled-Map-Navigation
This system enables hands-free interaction with digital maps, designed particularly for accessibility-focused or constrained environments. The solution integrates real-time gesture detection with a lightweight, fully offline map navigation interface.

Technology Stack used:
- Computer Vision: OpenCV for image processing
- Pose Estimation: MediaPipe for hand landmark detection
- Machine Learning: Scikit-learn for gesture classification / model training
- Web Mapping: OpenLayers (offline version) for rendering and map controls
- Backend & Integration: Python (FastAPI), HTML/JS for inter-process communication

System Features:
- Recognition of 7 custom gestures: move up, move down, move left, move right, zoom in, zoom out, stop.
- Real-time classification with visual feedback
- Gesture-to-command mapping via custom-trained ML model
- Responsive OpenLayers map functioning entirely offline
  

# How the code works:
- The project follows a client-server architecture with a camera-enabled frontend and a Python-based backend for gesture recognition:

Frontend (frontend.html):
- Uses getUserMedia() to access the webcam and display a live video stream.
- Captures a frame every 200ms using <canvas> and encodes it as a base64 JPEG.
- Sends the encoded image to the backend via an HTTP POST request (/process_frame).
- Receives the predicted gesture (e.g., "move up", "zoom in") and updates the map view using OpenLayers.
- Displays visual feedback (directional arrows, gesture label) and uses a crosshair for navigation reference.
  
Backend (bcknd.py)
- Built using FastAPI and runs a REST API server on localhost:8000.
- Receives image data from the frontend, decodes it using base64 and OpenCV.
- Passes the image to a pretrained gesture classification model (joblib) to predict the gesture.
- Sends the gesture label back as a JSON response.

A small demo:

https://github.com/user-attachments/assets/6d60908f-f6ad-4a48-89fc-4e4255b17c1a
