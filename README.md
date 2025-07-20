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

A small demo:
https://github.com/user-attachments/assets/6ad509b4-34ad-4a41-9ffc-4953f018230a
