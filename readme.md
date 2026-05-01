# 🎓 Exam Hall Cheating Detection System

A real-time AI-based cheating detection system using **YOLOv8 Pose Estimation + Computer Vision**, designed for **edge devices like NVIDIA Jetson Nano**.

---

## 🚀 Features

- Real-time student monitoring (18–22 FPS)
- Head movement detection (cheating behavior)
- Phone usage detection (object detection)
- Multi-person detection
- Privacy-preserving (no video storage)
- Live dashboard using Streamlit

---

## 🧠 System Architecture

Input → Preprocessing → YOLO Pose → Tracking → Rule Engine → Alerts

![Architecture](assets/architecture.png)

---

## 🔍 Models Used

| Model | Purpose |
|------|--------|
| YOLOv8n-pose | Human keypoints detection |
| YOLOv11n-pose | Comparison model |
| YOLOv8n | Phone detection |

---

## 📊 Results

| Metric | YOLOv8 | YOLOv11 |
|-------|--------|--------|
| FPS | 18–22 | 16–20 |
| Inference Time | 45–55 ms | 50–62 ms |
| Accuracy (AP50) | 72–78% | 70–76% |

📌 **Conclusion:** YOLOv8 performs better for edge deployment.

---

## 📸 Sample Output

### YOLOv8 Output
![YOLOv8](assets/yolov8_output.png)

### YOLOv11 Output
![YOLOv11](assets/yolov11_output.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Exam-Hall-Cheating-Detection.git
cd Exam-Hall-Cheating-Detection

pip install -r requirements.txt

**Project Output (Video)**:  
[Watch Demo Video](https://drive.google.com/file/d/1Xw_Etuy8kkIyVkNQYYfaReMpIsoFytDm/view?usp=drive_link) 

