Smart Motion Coach - Starter Repo


Overview
--------
This scaffold implements the pipeline for a video-only Smart Motion Coach with LLM-based coach tips. It contains modular stubs for each pipeline stage (pose, 3D lift, kinematics, retrieval, diagnosis, LLM prompt). Use this scaffold to plug-in production-grade models (HRNet, VideoPose3D, Video transformers, Groq API) and iterate.


Quick start
-----------
1. Create a virtual env and install requirements:
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt


2. Prepare models and datasets (see DATASETS section).
3. Run demo:
streamlit run app.py


Datasets
--------
- Human3.6M: for 3D lifting baselines
- PennAction / GolfDB: for sports-specific evaluation
- Ego-Exo4D: for feedback supervision (request access)


Model repos to integrate
------------------------
- HRNet 2D pose: https://github.com/leoxiaobin/HRNet
- VideoPose3D: https://github.com/facebookresearch/VideoPose3D
- VIBE: https://github.com/mkocabas/VIBE
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics


LLM / Groq
----------
- This scaffold contains a `feedback_llm.py` module with a template for calling the Groq API or another LLM. Store API keys in a .env file.