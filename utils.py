# utils.py
import cv2
import numpy as np
import tempfile
import os
import tarfile

def extract_archive_if_contains_pt(archive_path: str, dest_dir: str = "models"):
    """
    If archive contains .pt file(s) extract them to dest_dir.
    Archive path example: /mnt/data/archive.tar.gz
    """
    if not os.path.exists(archive_path):
        return []
    extracted = []
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pt"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=dest_dir)
                    extracted.append(os.path.join(dest_dir, member.name))
    except Exception:
        return []
    return extracted

def load_video_frames(file_like, max_frames=600):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(file_like.read())
    tmp.flush()
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames=[]
    count=0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count+=1
    cap.release()
    return frames, int(fps)

def choose_primary_person(kps_per_frame):
    """
    kps_per_frame: list of persons per frame -> [ [person1_kps, person2_kps,...], ...]
    We'll choose by maximal average confidence across frames:
    Returns array (T,17,3) for the selected person. If missing in a frame, fill zeros.
    """
    T = len(kps_per_frame)
    # gather all person ids by index across frames is hard; choose per-frame highest confidence person
    selected = []
    for frame_persons in kps_per_frame:
        if not frame_persons:
            # no detections - append zeros
            selected.append(np.zeros((17,3), dtype=float))
            continue
        # choose person with highest mean confidence
        best = None; best_score = -1
        for p in frame_persons:
            sc = np.mean(p[:,2])
            if sc > best_score:
                best_score = sc
                best = p
        selected.append(best)
    return np.stack(selected)  # (T,17,3)

def select_pose_by_bbox(frame_poses, shooter_bbox):
    """
    frame_poses: list of (17,3) poses
    shooter_bbox: (x1,y1,x2,y2)
    """
    if not frame_poses or shooter_bbox is None:
        return None

    x1, y1, x2, y2 = shooter_bbox
    best_pose = None
    best_iou = 0.0

    for p in frame_poses:
        xs = p[:, 0]
        ys = p[:, 1]

        px1, py1 = xs.min(), ys.min()
        px2, py2 = xs.max(), ys.max()

        # IoU
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_p = (px2 - px1) * (py2 - py1)
        area_s = (x2 - x1) * (y2 - y1)
        union = area_p + area_s - inter + 1e-6

        iou = inter / union

        if iou > best_iou:
            best_iou = iou
            best_pose = p

    return best_pose

