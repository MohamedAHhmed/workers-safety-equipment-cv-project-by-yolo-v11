# 🦺 Worker Safety Equipment Detection

A computer vision project to detect whether workers are wearing safety equipment such as helmets, vests, gloves, and boots using YOLOv8.

## 📸 Project Overview

This model automatically identifies whether workers are following safety protocols on construction sites or industrial environments by detecting:

- 🟡 **Helmet**
- 👕 **Reflective Vest**
- 🧤 **Gloves**
- 👟 **Boots**
- 🙅 **Non-Helmet (Not Wearing Helmet)**
- 🧍 **Person**
- ❌ **Bare Arms (No Long Sleeves)**

---

## 🧠 Model Details

We used [YOLOv11](https://github.com/ultralytics/ultralytics) for real-time object detection. The model was fine-tuned on the **[Worker Safety Equipment Dataset](https://universe.roboflow.com/wyhil-ru2ds/workers-safety-equipment-z1mra)** from Roboflow.

### ✅ Tools & Technologies:
- Python
- OpenCV
- YOLOv11 (Ultralytics)
- Roboflow Dataset
- Real-time webcam/video inference

---

## 🛠️ How to Run

### 1. Prepare the model

Make sure the trained model weights (`best.pt`) are in the working directory. You can train your own or use the one from Roboflow.

### 2. Run detection

Run the detection script:

```bash
python app.py
