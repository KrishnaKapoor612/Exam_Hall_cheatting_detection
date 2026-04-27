# inference.py
import cv2
import time


def run_inference(model, processed_frame, original_frame,
                  phone_detector=None,
                  session_id: str = None,
                  frame_number: int = 0,
                  timestamp_sec: float = 0.0):

    start_time = time.time()

    results = model.track(
        processed_frame,
        persist=True,
        conf=0.5,
        iou=0.7,
        tracker="bytetrack.yaml",
        imgsz=320,
        verbose=False
    )

    inference_time = time.time() - start_time
    fps            = 1.0 / max(inference_time, 1e-6)
    latency_ms     = inference_time * 1000

    # Phone detection
    phone_boxes = []
    if phone_detector:
        phone_boxes = phone_detector.detect_phones(processed_frame)

    from utils import rule_engine
    alerts    = rule_engine(results, original_frame.shape, phone_boxes)
    annotated = results[0].plot(conf=True, boxes=True, labels=True)

    # Draw phone boxes in orange
    for (x1, y1, x2, y2, conf_p) in phone_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(annotated, f"PHONE {conf_p:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Overlay metrics
    cv2.putText(annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated, f"Inference: {latency_ms:.1f} ms",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    for i, alert in enumerate(alerts):
        cv2.putText(annotated, f"ALERT: {alert['event']} (ID {alert['track_id']})",
                    (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return annotated, alerts, inference_time, fps