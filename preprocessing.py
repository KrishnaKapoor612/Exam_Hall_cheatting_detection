import cv2

from config import INFERENCE_SIZE

def preprocess_frame(frame, target_size=INFERENCE_SIZE):
    """
    Input data preparation function + Feature Extraction
    """
    # Resize while maintaining aspect ratio (letterbox style)
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Pad to square 640x640
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return padded