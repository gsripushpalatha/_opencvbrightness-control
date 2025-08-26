# 🖐️ Hand Gesture Brightness Control

This project uses **OpenCV**, **MediaPipe**, and **screen-brightness-control** to adjust your system brightness in real time based on the distance between your **thumb tip** and **index finger tip** captured through the webcam.

---

## ✨ Features
- Detects a single hand using **MediaPipe Hands**.
- Measures distance between thumb and index finger.
- Maps finger distance → system brightness (0–100%).
- Real-time webcam feed with visual feedback.
- Graceful fallback if brightness control is not supported on your system.

---

## 📦 Requirements

Make sure you have **Python 3.8+** installed.  
Then install the dependencies:

```bash
pip install opencv-python mediapipe screen-brightness-control numpy
