# ✋ Hand Sign Detection using MediaPipe and LSTM

This project implements a **real-time Hand Sign Detection system** using  
**MediaPipe Hands** for keypoint extraction and an **LSTM neural network** for gesture classification.  
Instead of using any external dataset, this system captures **your own gesture frames** using a webcam  
and uses them to train the LSTM model.

---

## 📌 Features
- Real-time hand landmark detection using **MediaPipe Hands**
- Extracts **21 keypoints (x, y, z)** per hand → 63 features/frame
- Custom dataset creation from **self-captured frames**
- LSTM model learns temporal patterns of gestures
- Accurate and lightweight model for real-time prediction

---

## 🛠 Technologies Used
- **Python 3.x**
- **MediaPipe Hands**
- **OpenCV**
- **NumPy**
- **TensorFlow / Keras**
- **Scikit-learn**

---

## 📂 Project Structure 
```
HandSignDetection/ 
│
├── MP_Data/ # Your self-captured sequences/
│ └── <gesture_name>/ # e.g., Hello, Yes, No/
│
├── action.h5 # Saved LSTM model/
|
├── Sign_Language_Detection.ipynb/
└── README.md
```

## 📸 How the System Works

### **1. Keypoint Extraction**
Using **MediaPipe Hands**, the system extracts:
- 21 Hand Landmarks  
Each with:
- **x, y, z coordinates**

So each frame =  21 landmarks × 3 values = 63 features


---

### **2. Sequence Creation**
Each gesture is represented as a sequence of **30 frames**: 30 frames × 63 features = Gesture sequence

These sequences are stored in the `Data/gesture_name/` folder.

---

### **3. LSTM Training**
An LSTM learns how the gesture evolves over time.

Input shape: (30 frames, 63 features)


---

## 🚀 Getting Started

### ✔ Install Required Libraries
```bash
pip install numpy opencv-python mediapipe tensorflow scikit-learn
