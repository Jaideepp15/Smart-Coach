Basketball Shooting Form Coach - 2024+ Implementation


Overview
--------
This project implements an AI-powered basketball shooting form analyzer. It uses state-of-the-art (2024+) models for 2D/3D pose estimation and kinematic analysis to provide actionable, coach-like feedback generated through an integrated LLM (via Groq API). The system identifies errors in shooting mechanics (release timing, elbow angle, knee flexion, hip extension) and offers corrective drills and motivation.


Latest Models Used (2024+)
---------------------------
- **Pose2D**: RTM-Pose (OpenMMLab, 2024) — High accuracy, real-time human pose estimation.
- **3D Pose**: HybrIK-X 2024 — Combines inverse kinematics with deep video modeling for robust sports motion capture.
- **Kinematics & Temporal Smoothing**: STCFormer (Spatial-Temporal Context Transformer, CVPR 2024) for smoother joint motion tracking.
- **LLM Feedback**: Groq API integration with prompt optimization for sports context.


Datasets
--------
- **BasketballDB-3D (2024)** — Annotated 3D shooting form dataset with key event frames (release, jump, apex, follow-through).
- **SportsPoseX (2024)** — Cross-sport 3D human motion dataset for temporal pose lifting benchmarks.
- **Ego-Exo4D (subset: basketball clips)** — Expert annotated sequences used for fine-tuning the feedback LLM.


Pipeline Summary
----------------
1. Input basketball shooting video (side/front angle)
2. 2D pose estimation (RTM-Pose 2024)
3. 3D pose lifting (HybrIK-X 2024)
4. Kinematic feature extraction (elbow, wrist, knee, hip)
5. Compare with expert reference library (BasketballDB-3D)
6. Diagnose deviations and timing mismatches
7. Generate actionable coaching feedback (Groq API)
8. Visualize skeleton overlay and corrected pose suggestions


Run Demo
--------
1. Install requirements: `pip install -r requirements.txt`
2. Add `.env` with `GROQ_API_KEY` and `GROQ_ENDPOINT`
3. Run `streamlit run app.py`
