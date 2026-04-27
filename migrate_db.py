"""
migrate_db.py
Run ONCE inside your project folder:  python migrate_db.py

Fixes the sqlite3.OperationalError: table sessions has no column named source
by migrating the old schema to the one expected by backend.py.
"""

import sqlite3
from pathlib import Path

DB_PATH = "exam_sessions.db"

def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── Check current columns ──────────────────────────────────────────────
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    print(f"Current sessions columns: {existing_cols}")

    # ── Add missing columns (safe — ALTER TABLE only if column absent) ─────
    additions = {
        "source":      "TEXT",
        "frames_dir":  "TEXT",
    }
    for col, coltype in additions.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {coltype}")
            print(f"  ✅ Added column: {col}")
        else:
            print(f"  ⏭  Column already exists: {col}")

    # ── Back-fill 'source' from 'video_path' if that old column exists ─────
    if "video_path" in existing_cols:
        conn.execute("UPDATE sessions SET source = video_path WHERE source IS NULL")
        print("  ✅ Back-filled 'source' from old 'video_path' column")

    # ── Ensure alerts table exists with correct schema ─────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,
            timestamp_s   REAL,
            frame_no      INTEGER,
            event_type    TEXT,
            track_id      INTEGER,
            confidence    REAL,
            snapshot_path TEXT
        );
    """)
    print("  ✅ Alerts table verified")

    conn.commit()
    conn.close()
    print(f"\n✅ Migration complete. '{DB_PATH}' is now compatible with backend.py")


if __name__ == "__main__":
    if not Path(DB_PATH).exists():
        print(f"'{DB_PATH}' not found — nothing to migrate (backend.py will create it fresh).")
    else:
        migrate()