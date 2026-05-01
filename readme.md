# 🎓 Exam Hall Cheating Detection System

A real-time AI-based cheating detection system using **YOLOv8 Pose Estimation + Computer Vision**, designed for **edge devices like NVIDIA Jetson Nano**.

---

## 🚀 Features

- Real-time student monitoring (14-20 FPS)
- Head movement detection (cheating behavior)
- Phone usage detection (object detection)
- Multi-person detection
- Privacy-preserving (no video storage)
- Live dashboard using Streamlit

---

## 🧠 System Architecture

Input → Preprocessing → YOLO Pose → Tracking → Rule Engine → Alerts

<img width="448" height="781" alt="Screenshot 2026-05-01 110751" src="https://github.com/user-attachments/assets/fa8f445a-a08c-4320-ab26-572f913f5f0a" />


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
<img width="392" height="398" alt="Screenshot 2026-05-01 101126" src="https://github.com/user-attachments/assets/71bd8504-9b20-4596-ad9f-9796f309c655" />


### YOLOv11 Output
<img width="446" height="532" alt="Screenshot 2026-05-01 101355" src="https://github.com/user-attachments/assets/b2d92ca2-1856-420b-819d-7b6ffc1cb7cc" />


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Exam-Hall-Cheating-Detection.git
cd Exam-Hall-Cheating-Detection

pip install -r requirements.txt

**Project Output (Video)**:  
[Watch Demo Video](https://drive.google.com/file/d/1Xw_Etuy8kkIyVkNQYYfaReMpIsoFytDm/view?usp=drive_link) 

