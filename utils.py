import math
from datetime import datetime
from config import HEAD_ANGLE_THRESHOLD, PROXIMITY_THRESHOLD


def calculate_head_angle(kpts):
    """
    Calculates horizontal head turn using ear + nose keypoints.
    COCO keypoints: 0=nose, 3=left_ear, 4=right_ear
    """
    if len(kpts) < 5:
        return 0.0

    nose  = kpts[0]
    l_ear = kpts[3]
    r_ear = kpts[4]

    if nose[2] < 0.3:
        return 0.0

    both_ears_visible  = l_ear[2] > 0.3 and r_ear[2] > 0.3
    only_left_visible  = l_ear[2] > 0.3 and r_ear[2] <= 0.3
    only_right_visible = r_ear[2] > 0.3 and l_ear[2] <= 0.3

    if both_ears_visible:
        mid_x    = (l_ear[0] + r_ear[0]) / 2
        ear_span = abs(r_ear[0] - l_ear[0])
        if ear_span == 0:
            return 0.0
        offset = abs(nose[0] - mid_x) / ear_span
        return offset * 90

    if only_left_visible or only_right_visible:
        return 60.0

    return 0.0


def phone_detector_check(phone_boxes, person_box):
    """
    Check if any detected phone box overlaps with this person's bounding box.
    Returns (True, confidence) if phone found near person, else (False, 0.0)
    """
    px1, py1, px2, py2 = person_box
    for (fx1, fy1, fx2, fy2, conf) in phone_boxes:
        overlap_x = fx1 < px2 and fx2 > px1
        overlap_y = fy1 < py2 and fy2 > py1
        if overlap_x and overlap_y:
            return True, conf
    return False, 0.0


def rule_engine(results, frame_shape, phone_boxes=None):
    """Rule-based malpractice detection engine"""
    alerts       = []
    phone_boxes  = phone_boxes or []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not results or len(results[0].boxes) == 0:
        return alerts

    tracks    = results[0].boxes
    keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else []

    # ── 0. Multiple Persons Detected ─────────────────────────────────────────
    total_persons = len(tracks)
    if total_persons > 1:
        alerts.append({
            "track_id" : -1,            # -1 = not tied to a single person
            "event"    : f"Multiple Persons Detected ({total_persons} people in frame)",
            "conf"     : 1.0,
            "timestamp": current_time
        })

    for i, box in enumerate(tracks):
        track_id = int(box.id.item()) if box.id is not None else i
        conf     = box.conf.item()
        kpts     = keypoints[i] if len(keypoints) > i else []

        if len(kpts) == 0:
            continue

        # ── 1. Head Turning ──────────────────────────────────────────────────
        angle = calculate_head_angle(kpts)
        if angle > HEAD_ANGLE_THRESHOLD:
            alerts.append({
                "track_id" : track_id,
                "event"    : "Head Turning Toward Neighbor",
                "conf"     : conf,
                "timestamp": current_time
            })

        # ── 2. Phone Usage (Object Detection Based) ──────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        phone_found, phone_conf = phone_detector_check(phone_boxes, (x1, y1, x2, y2))
        if phone_found:
            alerts.append({
                "track_id" : track_id,
                "event"    : "Phone Usage Detected",
                "conf"     : phone_conf,
                "timestamp": current_time
            })

    return alerts