# main.py
import cv2
import uuid
import time
from ultralytics import YOLO
from config import *
from preprocessing import preprocess_frame
from inference import run_inference
from logger import setup_logger
from phone_detector import PhoneDetector

# storage.py is replaced by backend.py
# If you want DB logging, run dashboard.py instead and use backend.start_session()

print(f"Using model: {MODEL_TYPE.upper()}")


def main():
    print("Initializing Exam Hall Malpractice Detection System.")

    model          = YOLO(MODEL_PATH)
    phone_detector = PhoneDetector()
    cap            = cv2.VideoCapture(INPUT_SOURCE)
    logger         = setup_logger()

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    if INPUT_SOURCE == 0:
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FORCE_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FORCE_RESOLUTION[1])
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera set to: {actual_w}x{actual_h} @ {actual_fps} FPS")
    else:
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video file FPS: {actual_fps}")

    print(f"System ready! Input: {'Webcam' if INPUT_SOURCE == 0 else INPUT_SOURCE}")

    frame_count = 0
    start_time  = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count  += 1
        timestamp_sec = time.time() - start_time
        small_frame   = cv2.resize(frame, (320, 320))

        annotated_frame, alerts, inf_time, fps = run_inference(
            model, small_frame, frame, phone_detector
        )

        for alert in alerts:
            logger.info(
                f"[{timestamp_sec:.1f}s] TRACK_{alert['track_id']} "
                f"- {alert['event']} | Conf: {alert['conf']:.2f}"
            )

        cv2.imshow("Exam Hall Malpractice Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total frames: {frame_count}. Alerts saved to {LOG_FILE}")
    print("Tip: Run 'streamlit run dashboard.py' for the full analytics dashboard.")


if __name__ == "__main__":
    main()