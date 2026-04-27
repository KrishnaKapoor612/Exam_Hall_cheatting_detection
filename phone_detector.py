# phone_detector.py
from ultralytics import YOLO
from config import DETECTION_MODEL_PATH, PHONE_CLASS_ID, PHONE_CONF_THRESHOLD


class PhoneDetector:
    def __init__(self):
        print("Loading object detection model for phone detection...")
        self.model = YOLO(DETECTION_MODEL_PATH)  # yolov8n.pt

    def detect_phones(self, frame):
        """
        Returns list of phone bounding boxes found in frame.
        Each box: (x1, y1, x2, y2, confidence)
        """
        results = self.model(frame, imgsz=320, verbose=False, conf=PHONE_CONF_THRESHOLD)
        phones  = []

        for box in results[0].boxes:
            cls  = int(box.cls.item())
            conf = box.conf.item()
            if cls == PHONE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                phones.append((x1, y1, x2, y2, conf))

        return phones

    def is_phone_near_person(self, phone_boxes, person_box):
        """
        Check if any detected phone overlaps or is near a person's bounding box.
        person_box: (x1, y1, x2, y2)
        """
        px1, py1, px2, py2 = person_box
        for (fx1, fy1, fx2, fy2, conf) in phone_boxes:
            # Check if phone box overlaps with person box
            overlap_x = fx1 < px2 and fx2 > px1
            overlap_y = fy1 < py2 and fy2 > py1
            if overlap_x and overlap_y:
                return True, conf
        return False, 0.0