# backend.py
"""
Central backend that:
  - Integrates with main.py, inference.py, phone_detector.py, config.py
  - Saves annotated frames + alert snapshots locally to disk
  - Maintains a real-time shared state dict (thread-safe)
  - Exposes start_session() / stop_session() used by dashboard.py
  - Persists session history to SQLite (exam_sessions.db)
"""

import cv2
import time
import uuid
import json
import sqlite3
import threading
import os
from pathlib import Path
from datetime import datetime
from collections import deque

# ── Lazy imports so backend.py works even if YOLO not installed yet ────────────
_yolo_available = False
try:
    from ultralytics import YOLO
    _yolo_available = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH          = "exam_sessions.db"
FRAMES_ROOT      = Path("saved_frames")       # locally saved frames
SNAPSHOTS_ROOT   = Path("alert_snapshots")    # saved on every alert
FRAMES_ROOT.mkdir(exist_ok=True)
SNAPSHOTS_ROOT.mkdir(exist_ok=True)

MAX_TIMELINE_PTS = 300   # keep last N seconds in the live buffer

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED REAL-TIME STATE  (written by detection thread, read by dashboard)
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

_state: dict = {
    # session info
    "session_id":    None,
    "running":       False,
    "started_at":    None,
    "source":        None,

    # live frame (JPEG bytes for dashboard display)
    "latest_frame_jpg": None,

    # rolling metrics (deques → converted to lists for dashboard)
    "fps_history":        deque(maxlen=MAX_TIMELINE_PTS),
    "latency_history":    deque(maxlen=MAX_TIMELINE_PTS),
    "alert_rate_history": deque(maxlen=MAX_TIMELINE_PTS),
    "head_angle_history": deque(maxlen=MAX_TIMELINE_PTS),
    "time_history":       deque(maxlen=MAX_TIMELINE_PTS),

    # counters
    "frame_count":      0,
    "total_alerts":     0,
    "phone_frames":     0,
    "away_frames":      0,
    "suspicious_frames":0,

    # last alert list (shown in live feed)
    "recent_alerts":  deque(maxlen=20),

    # risk score (live)
    "risk_score":  0.0,
    "risk_label":  "LOW",
}


def get_state() -> dict:
    """Return a safe snapshot of current state for the dashboard."""
    with _lock:
        return {
            "session_id":    _state["session_id"],
            "running":       _state["running"],
            "started_at":    _state["started_at"],
            "source":        _state["source"],
            "latest_frame_jpg": _state["latest_frame_jpg"],
            "fps_history":        list(_state["fps_history"]),
            "latency_history":    list(_state["latency_history"]),
            "alert_rate_history": list(_state["alert_rate_history"]),
            "head_angle_history": list(_state["head_angle_history"]),
            "time_history":       list(_state["time_history"]),
            "frame_count":        _state["frame_count"],
            "total_alerts":       _state["total_alerts"],
            "phone_frames":       _state["phone_frames"],
            "away_frames":        _state["away_frames"],
            "suspicious_frames":  _state["suspicious_frames"],
            "recent_alerts":      list(_state["recent_alerts"]),
            "risk_score":         _state["risk_score"],
            "risk_label":         _state["risk_label"],
        }


def _reset_state():
    with _lock:
        _state["session_id"]     = None
        _state["running"]        = False
        _state["started_at"]     = None
        _state["source"]         = None
        _state["latest_frame_jpg"] = None
        _state["fps_history"].clear()
        _state["latency_history"].clear()
        _state["alert_rate_history"].clear()
        _state["head_angle_history"].clear()
        _state["time_history"].clear()
        _state["frame_count"]      = 0
        _state["total_alerts"]     = 0
        _state["phone_frames"]     = 0
        _state["away_frames"]      = 0
        _state["suspicious_frames"]= 0
        _state["recent_alerts"].clear()
        _state["risk_score"] = 0.0
        _state["risk_label"] = "LOW"


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE  (lightweight — stores session history + alert log)
# ══════════════════════════════════════════════════════════════════════════════

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            started_at   TEXT,
            ended_at     TEXT,
            source       TEXT,
            total_frames INTEGER DEFAULT 0,
            total_alerts INTEGER DEFAULT 0,
            risk_score   REAL    DEFAULT 0.0,
            frames_dir   TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            timestamp_s  REAL,
            frame_no     INTEGER,
            event_type   TEXT,
            track_id     INTEGER,
            confidence   REAL,
            snapshot_path TEXT
        );
    """)
    conn.commit()
    conn.close()


_init_db()


def _db_create_session(sid, source, frames_dir):
    conn = _db()
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, started_at, source, frames_dir) VALUES (?,?,?,?)",
        (sid, datetime.utcnow().isoformat(), str(source), str(frames_dir))
    )
    conn.commit(); conn.close()


def _db_close_session(sid, total_frames, total_alerts, risk_score):
    conn = _db()
    conn.execute(
        "UPDATE sessions SET ended_at=?, total_frames=?, total_alerts=?, risk_score=? WHERE session_id=?",
        (datetime.utcnow().isoformat(), total_frames, total_alerts, risk_score, sid)
    )
    conn.commit(); conn.close()


def _db_log_alert(sid, ts, frame_no, event_type, track_id, conf, snap_path):
    conn = _db()
    conn.execute(
        "INSERT INTO alerts (session_id,timestamp_s,frame_no,event_type,track_id,confidence,snapshot_path) VALUES (?,?,?,?,?,?,?)",
        (sid, ts, frame_no, event_type, track_id, conf, str(snap_path))
    )
    conn.commit(); conn.close()


def list_sessions() -> list:
    conn = _db()
    rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_alerts(sid: str) -> list:
    conn = _db()
    rows = conn.execute(
        "SELECT * FROM alerts WHERE session_id=? ORDER BY timestamp_s", (sid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_session_record(sid: str):
    conn = _db()
    conn.execute("DELETE FROM alerts   WHERE session_id=?", (sid,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _calc_risk(phone_f, away_f, susp_f, total_f) -> tuple[float, str]:
    if total_f == 0:
        return 0.0, "LOW"
    raw   = (phone_f * 5 + susp_f * 2 + away_f * 1)
    score = min(100.0, raw / max(total_f, 1) * 100)
    if score < 20:   label = "LOW"
    elif score < 50: label = "MODERATE"
    elif score < 75: label = "HIGH"
    else:            label = "CRITICAL"
    return round(score, 1), label


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME SAVING
# ══════════════════════════════════════════════════════════════════════════════

def _save_frame(frame_bgr, session_dir: Path, frame_no: int, every_n: int = 30) -> str | None:
    """Save every Nth frame to disk. Returns path or None."""
    if frame_no % every_n != 0:
        return None
    path = session_dir / f"frame_{frame_no:06d}.jpg"
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return str(path)


def _save_snapshot(frame_bgr, snap_dir: Path, label: str, frame_no: int) -> str:
    """Save alert snapshot immediately."""
    fname = snap_dir / f"alert_{label}_{frame_no:06d}.jpg"
    cv2.imwrite(str(fname), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(fname)


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION THREAD
# ══════════════════════════════════════════════════════════════════════════════

_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _detection_loop(source, session_id: str, frames_dir: Path, snap_dir: Path,
                    save_every_n_frames: int, model_path: str):
    """Runs in a background thread. Writes to _state continuously."""
    global _state

    if not _yolo_available:
        with _lock:
            _state["running"] = False
        return

    # ── Load models ────────────────────────────────────────────────────────────
    try:
        from config import DETECTION_MODEL_PATH, PHONE_CLASS_ID, PHONE_CONF_THRESHOLD
        from phone_detector import PhoneDetector
        from utils import rule_engine

        pose_model     = YOLO(model_path)
        phone_detector = PhoneDetector()
    except Exception as e:
        print(f"[backend] Model load error: {e}")
        with _lock:
            _state["running"] = False
        return

    cap = cv2.VideoCapture(source if source != "webcam" else 0)
    if not cap.isOpened():
        print("[backend] Cannot open video source")
        with _lock:
            _state["running"] = False
        return

    frame_count  = 0
    start_t      = time.time()
    sec_alerts   = 0
    last_sec     = 0

    while not _stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        ts = time.time() - start_t
        cur_sec = int(ts)

        # Reset per-second alert counter
        if cur_sec != last_sec:
            with _lock:
                _state["alert_rate_history"].append(sec_alerts)
            sec_alerts = 0
            last_sec   = cur_sec

        small = cv2.resize(frame, (320, 320))

        # ── Inference ──────────────────────────────────────────────────────────
        t0 = time.time()
        results = pose_model.track(
            small, persist=True, conf=0.5, iou=0.7,
            tracker="bytetrack.yaml", imgsz=320, verbose=False
        )
        t1 = time.time()
        fps_val    = 1.0 / max(t1 - t0, 1e-6)
        latency_ms = (t1 - t0) * 1000

        phone_boxes = phone_detector.detect_phones(small)
        alerts      = rule_engine(results, frame.shape, phone_boxes)
        annotated   = results[0].plot(conf=True, boxes=True, labels=True)

        # Draw phone boxes
        for (x1, y1, x2, y2, conf_p) in phone_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated, f"PHONE {conf_p:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        # Overlay stats
        cv2.putText(annotated, f"FPS:{fps_val:.1f}  LAT:{latency_ms:.0f}ms  T:{ts:.0f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

        # ── Head angle ─────────────────────────────────────────────────────────
        head_angle = None
        try:
            import numpy as np
            kpts = results[0].keypoints
            if kpts is not None and kpts.xy is not None and len(kpts.xy) > 0:
                angles = []
                for kp in kpts.xy:
                    nose, le, re = kp[0], kp[1], kp[2]
                    if all(v > 0 for v in [*nose, *le, *re]):
                        mid  = (le + re) / 2
                        diff = nose - mid
                        angles.append(abs(float(np.degrees(np.arctan2(diff[1], diff[0])))))
                if angles:
                    head_angle = float(np.mean(angles))
        except Exception:
            pass

        # ── Save frame locally ─────────────────────────────────────────────────
        _save_frame(annotated, frames_dir, frame_count, save_every_n_frames)

        # ── Alert snapshots ────────────────────────────────────────────────────
        sec_alerts += len(alerts)
        for alert in alerts:
            snap = _save_snapshot(annotated, snap_dir,
                                  alert.get("event","UNK"), frame_count)
            _db_log_alert(
                session_id, ts, frame_count,
                alert.get("event","UNK"),
                alert.get("track_id", -1),
                alert.get("conf", 0.0),
                snap
            )

        # ── Encode frame as JPEG for dashboard ────────────────────────────────
        _, jpg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg_bytes  = jpg_buf.tobytes()

        # ── Update shared state ────────────────────────────────────────────────
        phone_det   = len(phone_boxes) > 0
        looking_away= any(a["event"] == "HEAD_TURN"       for a in alerts)
        suspicious  = any(a["event"] in ("PROXIMITY_ALERT","LEAN_FORWARD") for a in alerts)

        with _lock:
            _state["frame_count"]   = frame_count
            _state["latest_frame_jpg"] = jpg_bytes
            _state["fps_history"].append(fps_val)
            _state["latency_history"].append(latency_ms)
            _state["time_history"].append(ts)
            if head_angle is not None:
                _state["head_angle_history"].append(head_angle)

            if phone_det:   _state["phone_frames"]     += 1
            if looking_away:_state["away_frames"]       += 1
            if suspicious:  _state["suspicious_frames"] += 1
            _state["total_alerts"] += len(alerts)

            for a in alerts:
                _state["recent_alerts"].append({
                    "time": round(ts, 1),
                    "event": a.get("event","?"),
                    "track_id": a.get("track_id",-1),
                    "conf": round(a.get("conf",0.0), 2),
                })

            risk, rlabel = _calc_risk(
                _state["phone_frames"], _state["away_frames"],
                _state["suspicious_frames"], frame_count
            )
            _state["risk_score"] = risk
            _state["risk_label"] = rlabel

    cap.release()

    # ── Finalise session ───────────────────────────────────────────────────────
    with _lock:
        _db_close_session(
            session_id,
            _state["frame_count"],
            _state["total_alerts"],
            _state["risk_score"],
        )
        _state["running"] = False

    print(f"[backend] Session {session_id} ended. Frames saved → {frames_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (called by dashboard.py)
# ══════════════════════════════════════════════════════════════════════════════

def start_session(source="webcam",
                  save_every_n_frames: int = 30,
                  model_path: str = "yolo11n-pose.pt") -> str:
    """
    Start detection in a background thread.
    source: 0 / "webcam" for webcam, or a file path string.
    Returns session_id.
    """
    global _thread, _stop_event

    if _state["running"]:
        stop_session()

    _reset_state()
    _stop_event = threading.Event()

    session_id  = f"sess_{uuid.uuid4().hex[:8]}"
    frames_dir  = FRAMES_ROOT    / session_id
    snap_dir    = SNAPSHOTS_ROOT / session_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True,   exist_ok=True)

    src = 0 if source in (0, "webcam") else source

    _db_create_session(session_id, source, frames_dir)

    with _lock:
        _state["session_id"]  = session_id
        _state["running"]     = True
        _state["started_at"]  = datetime.now().strftime("%H:%M:%S")
        _state["source"]      = str(source)

    _thread = threading.Thread(
        target  = _detection_loop,
        args    = (src, session_id, frames_dir, snap_dir,
                   save_every_n_frames, model_path),
        daemon  = True,
    )
    _thread.start()
    return session_id


def stop_session():
    """Signal the detection thread to stop."""
    _stop_event.set()
    with _lock:
        _state["running"] = False