# config.py
MODEL_TYPE = "yolo11"
MODEL_PATH = "yolov8n-pose.pt" if MODEL_TYPE == "yolov8" else "yolo11n-pose.pt"

# ✅ Add this — standard object detection model
DETECTION_MODEL_PATH = "yolov8n.pt"  # downloads automatically, detects 80 COCO objects

INPUT_SOURCE = 0
HEAD_ANGLE_THRESHOLD = 40.0
PROXIMITY_THRESHOLD  = 180.0
LOG_FILE             = "alerts.log"
COMPARISON_LOG       = "model_comparison.log"
INFERENCE_SIZE       = 320
WARMUP_FRAMES        = 30
CAMERA_FPS           = 30
FORCE_RESOLUTION     = (640, 480)
PHONE_CLASS_ID       = 67   # COCO class 67 = cell phone
PHONE_CONF_THRESHOLD = 0.4  # confidence to count as phone detection