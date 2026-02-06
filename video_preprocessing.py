import cv2
import numpy as np
from collections import deque
from pose2d_yolo_pt import YOLOPosePT

# ---------------------------------------------------------
# 1. Decode video safely
# ---------------------------------------------------------
def decode_video(file_like, max_frames=600):
    import tempfile

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_like.read())
    tmp.flush()

    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, float(fps)


# ---------------------------------------------------------
# 2. FPS normalization (resample)
# ---------------------------------------------------------
def resample_fps(frames, src_fps, target_fps=30):
    if abs(src_fps - target_fps) < 1e-2:
        return frames, target_fps

    ratio = src_fps / target_fps
    idxs = (np.arange(0, len(frames)) / ratio).astype(int)
    idxs = idxs[idxs < len(frames)]
    return [frames[i] for i in idxs], target_fps


# ---------------------------------------------------------
# 3. Shooter detection + tracking (YOLO bbox)
# ---------------------------------------------------------
def detect_shooter_bboxes(frames, detector, conf_thresh=0.4):
    """
    Supports YOLO detectors that return:
    - (x1, y1, x2, y2, conf)
    - (x1, y1, x2, y2, conf, class_id)
    """

    bboxes = []
    prev = None

    for frame in frames:
        dets = detector.detect(frame)

        if not dets:
            bboxes.append(prev)
            continue

        # Normalize detections
        parsed = []
        for d in dets:
            if isinstance(d, dict):
                x1, y1, x2, y2 = d["bbox"]
                conf = d["conf"]
            else:
                # tuple-based detector
                x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
                conf = d[4]

            if conf >= conf_thresh:
                parsed.append((x1, y1, x2, y2, conf))

        if not parsed:
            bboxes.append(prev)
            continue

        # Choose primary shooter: largest bounding box
        best = max(
            parsed,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )

        bbox = tuple(map(int, best[:4]))
        prev = bbox
        bboxes.append(bbox)

    return bboxes

# ---------------------------------------------------------
# 4. Temporal bbox smoothing
# ---------------------------------------------------------
def smooth_bboxes(bboxes, window=7):
    smoothed = []
    q = deque(maxlen=window)

    for b in bboxes:
        if b is not None:
            q.append(b)
        if len(q) == 0:
            smoothed.append(None)
        else:
            arr = np.array(q)
            smoothed.append(tuple(arr.mean(axis=0).astype(int)))

    return smoothed


# ---------------------------------------------------------
# 5. Person-centric cropping + resize
# ---------------------------------------------------------
def crop_and_resize(frames, bboxes, out_size=512, pad=1.3):
    crops = []

    for frame, box in zip(frames, bboxes):
        if box is None:
            continue

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        side = int(max(x2-x1, y2-y1) * pad)

        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)

        crop = frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, (out_size, out_size))
        crops.append(crop)

    return crops


# ---------------------------------------------------------
# 6. Shot temporal segmentation (motion-based)
# ---------------------------------------------------------
def segment_shot_pose_guided(frames, pose_model, min_len=25, max_len=120):
    """
    Uses wrist/elbow vertical velocity to localize shot
    Falls back safely if detection is weak
    """

    # Run lightweight pose on downscaled frames (fast)
    small_frames = [cv2.resize(f, (256, 256)) for f in frames]
    kps = pose_model.predict_sequence(small_frames)

    if isinstance(kps, list):
        fixed = []
        for frame_kps in kps:
            if frame_kps is None:
                fixed.append(np.zeros((17, 3), dtype=np.float32))
                continue

            frame_kps = np.asarray(frame_kps)

            if frame_kps.shape != (17, 3):
                fixed.append(np.zeros((17, 3), dtype=np.float32))
            else:
                fixed.append(frame_kps)

        kps = np.stack(fixed, axis=0)

    if kps.ndim != 3 or kps.shape[1] < 11:
        return frames

    # Expect shape (T, J, 3)
    if kps is None or len(kps) < min_len:
        return frames  # fallback: do not trim

    # COCO indices (right hand dominant assumption)
    WRIST = 10
    ELBOW = 8

    y_wrist = kps[:, WRIST, 1]
    y_elbow = kps[:, ELBOW, 1]

    # Vertical velocity
    v_wrist = np.gradient(y_wrist)
    v_elbow = np.gradient(y_elbow)

    motion = np.abs(v_wrist) + 0.5 * np.abs(v_elbow)

    # Adaptive threshold
    thresh = np.percentile(motion, 85)
    idx = np.where(motion > thresh)[0]

    if len(idx) < min_len:
        return frames  # graceful fallback

    center = idx[len(idx) // 2]
    start = max(0, center - max_len // 2)
    end = min(len(frames), center + max_len // 2)

    return frames[start:end]


# ---------------------------------------------------------
# 7. Frame quality validation
# ---------------------------------------------------------
def validate_frames(frames, blur_thresh=80.0):
    valid = []

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur >= blur_thresh:
            valid.append(f)

    if len(valid) < 20:
        raise ValueError("Video too blurry / low quality")

    return valid


# ---------------------------------------------------------
# 8. Master preprocessing function
# ---------------------------------------------------------
def preprocess_video(
    uploaded_file,
    person_detector,
    target_fps=30,
    crop_size=512
):
    frames, fps = decode_video(uploaded_file)
    frames, fps = resample_fps(frames, fps, target_fps)

    bboxes = detect_shooter_bboxes(frames, person_detector)
    bboxes = smooth_bboxes(bboxes)

    crops = crop_and_resize(frames, bboxes, crop_size)
    model = YOLOPosePT(
        pt_path="models/yolo11n-pose.pt",
        device="cpu"   # segmentation pass should be cheap
    )
    crops = segment_shot_pose_guided(
        crops,
        pose_model=model
    )
    crops = validate_frames(crops)

    return crops, fps, bboxes
