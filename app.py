import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='🏀 Smart Shooting Coach (ONNX)', layout='wide')
st.title('🏀 Smart Basketball Shooting Coach — ONNX (Windows)')

from utils import load_video_frames, smart_crop_sequence
from detector import YOLODetector
from pose_estimation_onnx import RTMPoseONNX
from pose3d_hybrik_onnx import HybrIKONNX
from stc_smoother import STCFormerSmoother
from kinematics import Kinematics
from retrieval import ReferenceRetriever
from diagnosis import Diagnoser
from feedbackllm import CoachLLM
from visualization import Visualizer

MODEL_2D = os.getenv('RTMPOSE_ONNX', './models/rtmpose_l.onnx')
MODEL_3D = os.getenv('HYBRIK_ONNX', './models/hybrik.onnx')
DETECTOR = os.getenv('DETECTOR_MODEL', 'yolov8n.pt')
DEVICE = os.getenv('DEVICE', 'cpu')

uploaded = st.file_uploader('Upload shooting video (side/front, 8–15s recommended)', type=['mp4','mov','avi'])
if not uploaded:
    st.info('Upload a short clip (8–15s) of a jump-shot, side or 45° front view recommended.')
    st.stop()

frames, fps = load_video_frames(uploaded, max_frames=900)
st.info(f'Loaded {len(frames)} frames @ {fps} fps')

if st.button('Show preview'):
    st.video(uploaded)

level = st.selectbox('Experience level', ['beginner','intermediate','advanced'])

with st.spinner('Detecting persons & cropping...'):
    detector = YOLODetector(model_path=DETECTOR, device=DEVICE)
    crops, boxes = smart_crop_sequence(frames, detector)

with st.spinner('Running 2D pose (RTMPose-L ONNX)...'):
    pose2d = RTMPoseONNX(MODEL_2D, device=DEVICE)
    kps2d = pose2d.predict_sequence(crops)

with st.spinner('Lifting to 3D (HybrIK ONNX or fallback)...'):
    lifter = HybrIKONNX(MODEL_3D, device=DEVICE)
    kps3d = lifter.lift_sequence(kps2d)

with st.spinner('Temporal smoothing...'):
    smoother = STCFormerSmoother(kernel_size=5)
    kps3d_sm = smoother.smooth(kps3d)

with st.spinner('Extracting kinematics...'):
    kin = Kinematics()
    features = kin.extract_features(kps3d_sm, fps=fps)

with st.spinner('Retrieving reference...'):
    retriever = ReferenceRetriever(index_path=os.getenv('REF_FAISS_INDEX', None))
    ref_clip, ref_feats = retriever.query(features)

with st.spinner('Diagnosing...'):
    diagnoser = Diagnoser()
    issues = diagnoser.diagnose(features, ref_feats)

with st.spinner('Generating coach feedback...'):
    coach = CoachLLM(groq_endpoint=os.getenv('GROQ_ENDPOINT'), groq_key=os.getenv('GROQ_API_KEY'))
    tip = coach.generate_tip('basketball_shot', level, issues, features, ref_context=ref_clip.get('meta'))

viz = Visualizer()
overlay_path = viz.create_overlay(frames, kps2d, ref_clip=ref_clip)

st.success('✅ Analysis complete!')
st.subheader('🏀 Coach Feedback')
st.write(tip['text'])

st.subheader('🔍 Diagnosed Issues')
st.json(issues)

st.subheader('📹 Overlay Comparison')
st.video(overlay_path)
