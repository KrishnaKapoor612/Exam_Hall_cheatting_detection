# Exam Hall Malpractice Detection System using YOLOv8 Pose Estimation

## 1. Project Title
Exam Hall Malpractice Detection System using YOLOv8 Pose Estimation

## 2. Problem Statement
Academic malpractice (head-turning to copy, phone usage under desk, paper passing) undermines examination integrity. Manual invigilation is prone to fatigue and limited coverage. Existing solutions either require cloud connectivity (privacy & latency issues) or only perform basic object detection without understanding behavioral context between students.

This project solves the problem by building a **real-time, privacy-preserving, on-device** behavior analysis system using pose keypoints and tracking — deployed entirely on affordable edge hardware (NVIDIA Jetson Nano).

## 3. Role of Edge Computing
- **Components running on Jetson Nano**: YOLOv8 Pose inference, DeepSORT-style tracking (via Ultralytics tracker), rule engine for spatial-temporal analysis, alert logging, and live annotated output.
- **Justification for edge over cloud**: Exam halls often have no/restricted internet. Cloud solutions introduce latency, bandwidth cost, and privacy risks (raw video transmission). Edge computing ensures **zero network dependency**, sub-50 ms inference, and **no raw video ever stored** — only timestamped alerts.
- **Benefits**: Real-time alerts, full offline operation, privacy compliance, low cost (~₹8,000 hardware).

## 4. Methodology / Approach
**Overall Pipeline**:  
**Input** → **Preprocessing** → **Model Inference + Tracking** → **Rule Engine** → **Output (annotated frame + alert log)**

- **Input**: Camera feed (webcam / video).
- **Preprocessing**: Frame resizing and normalization.
- **Model**: YOLOv8 Pose + built-in tracker for persistent student IDs.
- **Rule Engine**: Analyzes head orientation, wrist position, and inter-student proximity to detect malpractice.
- **Output**: Live annotated video with FPS/inference time + alerts logged to `alerts.log`.

## 5. Model Details
- **Type**: YOLOv8 Pose Estimation (nano variant — optimized for edge).
- **Architecture**: CSPDarknet backbone + PANet neck + Pose head (17 COCO keypoints).
- **Input size and format**: 640×640 RGB image.
- **Framework**: Ultralytics (PyTorch).
- **Optimization**: Supports TensorRT export on Jetson Nano (optional for 2× speedup).

## 6. Training Details
- **Dataset used**: Pretrained on COCO keypoints + fine-tuned on custom exam-hall images (annotated with CVAT).
- **Training procedure**: Transfer learning from `yolov8n-pose.pt` using Ultralytics trainer (AdamW optimizer, cosine LR schedule, 50 epochs).
- **Performance graphs**:
  - Loss vs Epoch: `graphs/loss.png` (generated automatically by `training.py`)
  - Accuracy vs Epoch: `graphs/accuracy.png` (generated automatically by `training.py`)

**Project Output (Video)**:  
[Watch Demo Video](project_output.mp4) *(record your own demo on Jetson Nano / laptop and place the file in the repo root)*

**Project Models Used**:  
- `yolov8n-pose.pt` (automatically downloaded on first run)
- Training graphs are saved in `/graphs/` folder after running `training.py`