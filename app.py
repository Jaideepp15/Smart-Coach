# app.py
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import cv2

from utils import load_video_frames, extract_archive_if_contains_pt, choose_primary_person, select_pose_by_bbox
from detector import YOLODetector
from pose2d_yolo_pt import YOLOPosePT
from pose3d_ncret import NCRETNetLifter
from stc_smoother import STCFormerSmoother
from kinematics import Biomechanics
from visualization import Visualizer
from feedbackllm import generate_coaching
from video_preprocessing import preprocess_video

st.set_page_config(page_title="Smart Coach", layout="wide")
st.title("🏀 AI Powered Basketball Shooting Coach")

MODEL_PT_DEFAULT = os.getenv("YOLO_PT","models/yolo11n-pose.pt")
LIFTER_MODEL = os.getenv("DEFAULT",r"models\PoseRetNet_h36m_cpn_243f.pth")
DEVICE = os.getenv("DEVICE","cpu")

if "frames" not in st.session_state:
    st.session_state.frames = None
    st.session_state.fps = None
    st.session_state.video_id = None
    st.session_state.shooter_bboxes  = None

st.markdown("Upload a short jump-shot clip (8-15s recommended).")

uploaded = st.file_uploader("Upload video", type=['mp4','mov','avi'])
if not uploaded:
    st.info("Upload a video to start.")
    st.stop()

video_id = uploaded.name + str(uploaded.size)

if st.session_state.video_id != video_id:
    with st.spinner("Loading video frames..."):
        detector = YOLODetector(device=DEVICE)
        frames, fps, bboxes = preprocess_video(
            uploaded_file=uploaded,
            person_detector=detector,
            target_fps=30,
            crop_size=512
        )
        st.session_state.frames = frames
        st.session_state.fps = fps
        st.session_state.video_id = video_id
        st.session_state.shooter_bboxes  = bboxes

    st.success(f"Loaded {len(frames)} frames @ {fps} FPS")

st.subheader("Shooter Profile")

handedness = st.radio(
    "What is your shooting hand?",
    options=["Right-handed", "Left-handed"],
    index=0,
    horizontal=True,
    help="This helps the Smart Coach interpret elbow, wrist, and shoulder mechanics correctly."
)

miss_options = ["left", "right", "short", "long"]

user_misses = st.multiselect(
    "How are your shots typically missing?",
    options=miss_options,
    help="Select all miss-directions that apply. This helps the Smart Coach personalize feedback."
)

# Require at least one miss pattern
if len(user_misses) == 0:
    st.warning("Please select at least one miss pattern before running analysis.")
    run_enabled = False
else:
    run_enabled = True

if st.button("Run analysis", disabled=not run_enabled):
    frames = st.session_state.frames
    fps = st.session_state.fps
    with st.spinner("Running 2D pose detection..."):
        pose2d = YOLOPosePT(MODEL_PT_DEFAULT, device=DEVICE)
        # process frames; pick primary person per frame
        kps_per_frame = pose2d.predict_sequence(frames)
        # choose primary person across frames
        mapped=choose_primary_person(kps_per_frame)
    st.success("2D pose done.")

    with st.spinner("3D lifting..."):
        lifter = NCRETNetLifter(LIFTER_MODEL, device=DEVICE)
        # lifter expects (T,17,2) or (B,T,J,2) - adapt:
        inp = mapped[..., :2]  # (T,17,2)
        out3d = lifter.lift(inp)  # (T,17,3)
    st.success("3D lift done.")

    with st.spinner("Smoothing & analysis..."):
        smoother = STCFormerSmoother()
        out3d_sm = smoother.smooth(out3d)
        kin = Biomechanics()
        biomech = kin.extract_biomechanics(out3d_sm, fps,shooting_hand=handedness)
    st.success("Analysis done.")

    viz = Visualizer()

    # create final overlay video
    overlay = viz.create_overlay(
        frames=frames,
        kps2d=mapped,
        output_path="analysis_overlay.mp4",
        fps=fps,
        dip_frame=biomech["dip_frame"],
        extension_frame=biomech["extension_frame"],
        release_frame=biomech["release_frame"]
    )

    st.video(overlay)

    # Display Biomechanics
    st.subheader("📘 Biomechanical Breakdown")

    col1, col2, col3 = st.columns(3)

    # -------- Column 1: Key Frames --------
    col1.metric("Release frame", biomech["release_frame"])
    col1.metric("Dip frame", biomech["dip_frame"])
    col1.metric("Extension frame", biomech["extension_frame"])

    # -------- Column 2: Angles at Release --------
    angles_rel = biomech["angles_at_release"]
    col2.metric("Elbow @ release", f"{angles_rel['elbow']:.1f}°")
    col2.metric("Knee @ release", f"{angles_rel['knee']:.1f}°")
    col2.metric("Shoulder @ release", f"{angles_rel['shoulder']:.1f}°")

    # -------- Column 3: Timing & Coordination --------
    timing = biomech["timing"]
    coord = biomech["coordination"]

    col3.metric("Dip duration", f"{timing['dip_duration_s']:.2f}s")
    col3.metric("Drive duration", f"{timing['drive_duration_s']:.2f}s")
    col3.metric("Knee → Elbow delay", f"{coord['knee_to_elbow_delay_s']*1000:.0f} ms")

    st.markdown("---")

    # =========================
    # Advanced Metrics
    # =========================
    st.subheader("🔬 Advanced Motion Metrics")

    col4, col5 = st.columns(2)

    rom = biomech["range_of_motion"]
    cons = biomech["consistency"]

    col4.metric("Elbow ROM", f"{rom['elbow_extension']:.1f}°")
    col4.metric("Knee ROM", f"{rom['knee_extension']:.1f}°")

    col5.metric("Elbow smoothness", f"{cons['elbow_smoothness']:.2f}")
    col5.metric("Knee smoothness", f"{cons['knee_smoothness']:.2f}")


    with st.spinner("Generating coaching with Groq..."):
        try:
            miss_dir=",".join(user_misses)
            coach = generate_coaching(biomech, miss_dir, handedness=handedness)
            st.success("Coach advice generated.")
            st.markdown("#### Coach's feedback")
            st.write(coach["coach_text"])
            # optionally show the prompt for debugging
            with st.expander("Prompt (debug)"):
                st.code(coach["prompt"])
        except Exception as e:
            st.error(f"LLM coaching failed: {e}")
    st.success("Done!")
