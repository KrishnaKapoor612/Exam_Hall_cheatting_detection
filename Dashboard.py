# dashboard.py
import time
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import backend

st.set_page_config(
    page_title="ExamGuard · Live",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

*, html, body { box-sizing: border-box; }
.stApp { background: #080c12; }
[data-testid="stSidebar"] { background: #0b1018 !important; border-right: 1px solid #1a2535; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; color:#e2e8f0 !important; letter-spacing:.03em; }
p, li { color:#64748b; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }

.kpi {
    background: #0d1420;
    border: 1px solid #1a2d45;
    border-top: 3px solid var(--c, #3b82f6);
    border-radius: 6px;
    padding: 14px 18px 12px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: var(--c, #3b82f6); line-height: 1; margin-bottom: 4px;
}
.kpi-lbl {
    font-size: .62rem; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase; color: #334155;
}
.risk-pill {
    display:inline-block; padding: 5px 20px; border-radius: 4px;
    font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:1.1rem;
    letter-spacing:.08em; border: 1px solid var(--rc);
    background: color-mix(in srgb, var(--rc) 12%, transparent); color: var(--rc);
}
.alert-item {
    display:flex; gap:10px; align-items:center;
    padding: 7px 10px; background:#0d1420;
    border-left: 3px solid #ef4444; border-radius: 4px;
    margin-bottom: 5px; font-family:'IBM Plex Mono',monospace;
    font-size:.72rem; color:#cbd5e1;
}
.alert-badge {
    background:#ef444422; color:#ef4444;
    padding:2px 8px; border-radius:3px; font-weight:700; white-space:nowrap;
}
.sec-lbl {
    font-family:'IBM Plex Mono',monospace; font-size:.6rem;
    letter-spacing:.18em; text-transform:uppercase; color:#1d4ed8;
    border-bottom:1px solid #1a2d45; padding-bottom:5px; margin-bottom:10px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.live-dot {
    display:inline-block; width:9px; height:9px; border-radius:50%;
    background:#ef4444; animation: pulse 1.2s infinite; margin-right:7px;
}
.alert-scroll { max-height:240px; overflow-y:auto; }
.stButton>button {
    background:#1d4ed8 !important; color:#fff !important;
    border:none !important; border-radius:5px !important;
    font-family:'IBM Plex Mono',monospace !important;
    font-weight:600 !important; letter-spacing:.05em !important;
}
.stButton>button:hover { background:#2563eb !important; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout (NO xaxis/yaxis here — set per chart) ─────────────────
PL_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(13,20,32,0.9)",
    font=dict(family="IBM Plex Mono", color="#475569", size=10),
    margin=dict(l=36, r=12, t=32, b=32),
)

GRID_STYLE = dict(gridcolor="#1a2535", zerolinecolor="#1a2535", tickcolor="#1a2535")

C = dict(blue="#3b82f6", red="#ef4444", amber="#f59e0b",
         green="#22c55e", purple="#a855f7", cyan="#06b6d4", orange="#f97316")

RISK_COLORS = {"LOW": C["green"], "MODERATE": C["amber"],
               "HIGH": C["red"],  "CRITICAL": "#b91c1c"}


def kpi_html(value, label, color="#3b82f6"):
    return (f"<div class='kpi' style='--c:{color}'>"
            f"<div class='kpi-val'>{value}</div>"
            f"<div class='kpi-lbl'>{label}</div></div>")


def mini_line(y_data, color, title, height=160):
    fig = go.Figure(go.Scatter(
        y=y_data, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace("#","") and f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
    ))
    fig.update_layout(**PL_BASE, height=height,
                      title=dict(text=title, font=dict(size=11, color="#64748b")),
                      xaxis=dict(**GRID_STYLE),
                      yaxis=dict(**GRID_STYLE))
    return fig


def risk_from_score(s):
    if s < 20:  return "LOW"
    if s < 50:  return "MODERATE"
    if s < 75:  return "HIGH"
    return "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:Barlow Condensed,sans-serif;font-size:1.6rem;"
        "font-weight:800;color:#e2e8f0;letter-spacing:.04em'>🎓 EXAMGUARD</div>"
        "<div style='font-size:.65rem;color:#334155;letter-spacing:.12em;"
        "text-transform:uppercase;margin-bottom:16px'>Malpractice Detection</div>",
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio("Navigation", ["🔴  Live Monitor", "📁  Session History"])
    st.divider()

    st.markdown("<div class='sec-lbl'>Detection Controls</div>", unsafe_allow_html=True)

    src_type = st.radio("Source", ["Webcam", "Video file"], horizontal=True)
    vid_path_str = None

    if src_type == "Video file":
        uploaded = st.file_uploader("Video file", type=["mp4","avi","mov","mkv"])
        if uploaded:
            save_dir = Path("uploaded_videos")
            save_dir.mkdir(exist_ok=True)
            vid_path_str = str(save_dir / uploaded.name)
            Path(vid_path_str).write_bytes(uploaded.read())
            st.caption(f"✅ {uploaded.name}")

    save_n = st.slider("Save frame every N frames", 3, 120, 30)

    col_a, col_b = st.columns(2)
    start_clicked = col_a.button("▶ START", use_container_width=True)
    stop_clicked  = col_b.button("⏹ STOP",  use_container_width=True)

    if start_clicked:
        source = vid_path_str if src_type == "Video file" and vid_path_str else "webcam"
        try:
            from config import MODEL_PATH
        except ImportError:
            MODEL_PATH = "yolo11n-pose.pt"
        sid = backend.start_session(source=source, save_every_n_frames=save_n, model_path=MODEL_PATH)
        st.session_state["active_sid"] = sid
        st.success(f"Started\n`{sid}`")

    if stop_clicked:
        backend.stop_session()
        st.warning("Session stopped.")

    st.divider()
    state_peek = backend.get_state()
    if state_peek["session_id"]:
        fdir = backend.FRAMES_ROOT / state_peek["session_id"]
        st.caption(f"📂 Frames → `{fdir}`")
        snap_dir = backend.SNAPSHOTS_ROOT / state_peek["session_id"]
        st.caption(f"📸 Snapshots → `{snap_dir}`")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔴  Live Monitor":

    state   = backend.get_state()
    running = state["running"]

    status_html = (
        "<span class='live-dot'></span>"
        "<span style='color:#ef4444;font-family:IBM Plex Mono;font-weight:700;font-size:.8rem'>LIVE</span>"
        if running else
        "<span style='color:#334155;font-family:IBM Plex Mono;font-size:.8rem'>● IDLE</span>"
    )
    rc = RISK_COLORS.get(state["risk_label"], C["green"])

    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:4px'>"
        f"<h1 style='margin:0;font-size:2.2rem'>LIVE MONITOR</h1>"
        f"<div>{status_html}</div></div>"
        f"<div style='font-size:.7rem;color:#334155;font-family:IBM Plex Mono'>"
        f"Session: <span style='color:#475569'>{state['session_id'] or '—'}</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Video feed + KPIs ─────────────────────────────────────────────────────
    vid_col, kpi_col = st.columns([3, 2])

    with vid_col:
        st.markdown("<div class='sec-lbl'>Live Feed</div>", unsafe_allow_html=True)
        if state["latest_frame_jpg"]:
            st.image(state["latest_frame_jpg"], channels="BGR",
                     use_column_width=True, caption="Annotated detection output")
        else:
            st.markdown(
                "<div style='background:#0d1420;border:1px dashed #1a2d45;border-radius:6px;"
                "height:320px;display:flex;align-items:center;justify-content:center;"
                "color:#1e3a5f;font-family:IBM Plex Mono;font-size:.9rem'>"
                "▶ Start session to see feed</div>",
                unsafe_allow_html=True
            )

    with kpi_col:
        st.markdown("<div class='sec-lbl'>Real-Time Metrics</div>", unsafe_allow_html=True)

        fps_val = state["fps_history"][-1]     if state["fps_history"]     else 0
        lat_val = state["latency_history"][-1] if state["latency_history"] else 0
        ha_val  = state["head_angle_history"][-1] if state["head_angle_history"] else 0

        st.markdown(
            f"<div style='margin-bottom:12px'>"
            f"<span class='risk-pill' style='--rc:{rc}'>"
            f"⚠ {state['risk_label']}  ·  {state['risk_score']:.1f}/100"
            f"</span></div>",
            unsafe_allow_html=True
        )

        r1a, r1b, r1c = st.columns(3)
        r2a, r2b, r2c = st.columns(3)
        r1a.markdown(kpi_html(f"{fps_val:.0f}",     "FPS",        C["green"]),  unsafe_allow_html=True)
        r1b.markdown(kpi_html(f"{lat_val:.0f}ms",   "Latency",    C["purple"]), unsafe_allow_html=True)
        r1c.markdown(kpi_html(state["frame_count"], "Frames",     C["blue"]),   unsafe_allow_html=True)
        r2a.markdown(kpi_html(state["total_alerts"],"Alerts",     C["red"]),    unsafe_allow_html=True)
        r2b.markdown(kpi_html(f"{ha_val:.0f}°",     "Head Angle", C["cyan"]),   unsafe_allow_html=True)
        r2c.markdown(kpi_html(state["phone_frames"],"Phone Det.", C["orange"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Posture stacked bar
        total    = max(state["frame_count"], 1)
        norm_pct = max(0, total - state["away_frames"] - state["suspicious_frames"]) / total * 100
        away_pct = state["away_frames"]       / total * 100
        susp_pct = state["suspicious_frames"] / total * 100

        fig_bar = go.Figure()
        for lbl, val, col in [("Normal", norm_pct, C["green"]),
                               ("Away",   away_pct,  C["amber"]),
                               ("Suspicious", susp_pct, C["red"])]:
            fig_bar.add_trace(go.Bar(name=lbl, x=[val], y=["Posture"],
                                     orientation="h", marker_color=col,
                                     text=f"{val:.0f}%", textposition="inside"))
        fig_bar.update_layout(
            **PL_BASE,
            height=90,
            barmode="stack",
            showlegend=True,
            legend=dict(orientation="h", y=1.6, x=0),
            xaxis=dict(range=[0, 100], ticksuffix="%", **GRID_STYLE),
            yaxis=dict(**GRID_STYLE),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Timeline charts ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    for col, data, color, title in [
        (c1, state["fps_history"],        C["green"],  "FPS"),
        (c2, state["latency_history"],     C["purple"], "Latency (ms)"),
        (c3, state["alert_rate_history"],  C["red"],    "Alerts / sec"),
        (c4, state["head_angle_history"],  C["cyan"],   "Head angle (°)"),
    ]:
        with col:
            if data:
                st.plotly_chart(mini_line(data, color, title),
                                use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.markdown(
                    f"<div style='height:160px;border:1px dashed #1a2d45;border-radius:6px;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"color:#1e3a5f;font-family:IBM Plex Mono;font-size:.7rem'>{title}</div>",
                    unsafe_allow_html=True
                )

    # ── Alert log ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='sec-lbl'>Recent Alerts</div>", unsafe_allow_html=True)
    alerts = state["recent_alerts"]
    if not alerts:
        st.markdown("<div style='color:#1e3a5f;font-family:IBM Plex Mono;font-size:.8rem'>No alerts yet.</div>",
                    unsafe_allow_html=True)
    else:
        html_rows = ""
        for a in reversed(list(alerts)):
            html_rows += (
                f"<div class='alert-item'>"
                f"<span style='color:#334155;min-width:52px'>{a['time']}s</span>"
                f"<span class='alert-badge'>{a['event']}</span>"
                f"<span>Track <b>{a['track_id']}</b></span>"
                f"<span style='color:#475569'>conf {a['conf']}</span>"
                f"</div>"
            )
        st.markdown(f"<div class='alert-scroll'>{html_rows}</div>", unsafe_allow_html=True)

    # Auto-refresh
    if running:
        time.sleep(0.5)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📁  Session History":

    st.markdown("# SESSION HISTORY")
    st.markdown("---")

    sessions = backend.list_sessions()

    if not sessions:
        st.info("No completed sessions yet.")
        st.stop()

    df = pd.DataFrame(sessions)
    if "risk_score" in df.columns:
        df["Risk"] = df["risk_score"].apply(risk_from_score)

    show_cols = [c for c in ["session_id","started_at","ended_at","source",
                              "total_frames","total_alerts","risk_score","Risk"] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=260)
    st.markdown("---")

    sids = [s["session_id"] for s in sessions]
    sel  = st.selectbox("Drill into session", sids)

    if sel:
        sess        = next((s for s in sessions if s["session_id"] == sel), {})
        alerts_list = backend.get_session_alerts(sel)

        rc_score = sess.get("risk_score", 0.0)
        rc_label = risk_from_score(rc_score)
        rc_color = RISK_COLORS[rc_label]

        h1, h2, h3, h4 = st.columns(4)
        h1.markdown(kpi_html(sess.get("total_frames", 0), "Frames",     C["blue"]),   unsafe_allow_html=True)
        h2.markdown(kpi_html(sess.get("total_alerts", 0), "Alerts",     C["red"]),    unsafe_allow_html=True)
        h3.markdown(kpi_html(f"{rc_score:.1f}",            "Risk Score", rc_color),   unsafe_allow_html=True)
        h4.markdown(kpi_html(len(alerts_list),             "Events",     C["amber"]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if alerts_list:
            alerts_df = pd.DataFrame(alerts_list)

            ec = alerts_df["event_type"].value_counts().reset_index()
            ec.columns = ["Event", "Count"]
            fig_ec = px.bar(ec, x="Count", y="Event", orientation="h",
                            color_discrete_sequence=[C["red"]])
            fig_ec.update_layout(**PL_BASE, height=220,
                                 title="Alert Event Types",
                                 xaxis=dict(**GRID_STYLE),
                                 yaxis=dict(**GRID_STYLE))
            st.plotly_chart(fig_ec, use_container_width=True)

            fig_tl = px.scatter(alerts_df, x="timestamp_s", y="event_type",
                                color="event_type",
                                color_discrete_sequence=list(C.values()))
            fig_tl.update_layout(**PL_BASE, height=220,
                                 title="Alert Timeline",
                                 showlegend=False,
                                 xaxis=dict(title="Time (s)", **GRID_STYLE),
                                 yaxis=dict(**GRID_STYLE))
            st.plotly_chart(fig_tl, use_container_width=True)

            show_c = [c for c in ["timestamp_s","frame_no","event_type",
                                   "track_id","confidence","snapshot_path"]
                      if c in alerts_df.columns]
            st.dataframe(alerts_df[show_c], use_container_width=True, height=240)

            csv = alerts_df.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv,
                               file_name=f"alerts_{sel}.csv", mime="text/csv")

            # Snapshot preview
            if "snapshot_path" in alerts_df.columns:
                valid_snaps = [p for p in alerts_df["snapshot_path"].dropna().tolist()
                               if p and os.path.exists(p)]
                if valid_snaps:
                    st.markdown("<div class='sec-lbl'>Alert Snapshots (last 8)</div>",
                                unsafe_allow_html=True)
                    img_cols = st.columns(4)
                    for i, sp in enumerate(valid_snaps[-8:]):
                        row = alerts_df[alerts_df["snapshot_path"] == sp].iloc[0]
                        with img_cols[i % 4]:
                            st.image(sp, use_column_width=True,
                                     caption=f"{row['event_type']} @ {row['timestamp_s']:.1f}s")
        else:
            st.success("No alert events in this session.")

        # Saved frames preview
        fdir = backend.FRAMES_ROOT / sel
        if fdir.exists():
            frame_files = sorted(fdir.glob("*.jpg"))
            st.markdown("---")
            st.markdown(f"<div class='sec-lbl'>Saved Frames — {len(frame_files)} images in `{fdir}`</div>",
                        unsafe_allow_html=True)
            if frame_files:
                prev_cols = st.columns(5)
                for i, fp in enumerate(frame_files[-5:]):
                    with prev_cols[i % 5]:
                        st.image(str(fp), use_column_width=True, caption=fp.name)

        st.markdown("---")
        if st.button(f"🗑 Delete session `{sel}`"):
            backend.delete_session_record(sel)
            import shutil
            for d in [backend.FRAMES_ROOT / sel, backend.SNAPSHOTS_ROOT / sel]:
                if d.exists():
                    shutil.rmtree(d)
            st.success("Deleted.")
            st.rerun()