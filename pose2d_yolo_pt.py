# pose2d_yolo_pt.py
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple

class YOLOPosePT:
    def __init__(self, pt_path: str = "models/yolo11n-pose.pt", device: str = "cpu"):
        """
        pt_path: local .pt Ultralytics checkpoint containing pose model (yolo11n-pose.pt)
        device: 'cpu' or 'cuda'
        """
        self.pt_path = pt_path
        self.device = device
        self.model = YOLO(pt_path)  # Ultralytics will detect task type (pose)
        # Force device if provided
        try:
            self.model.to(device)
        except Exception:
            pass

    def predict_frame(self, img_rgb: np.ndarray, conf: float = 0.25):
        """
        img_rgb: H,W,3 (RGB)
        returns list of keypoints per detected person:
        [ (K,3), (K,3), ... ] where K ~17 (COCO order) and coords are in original image pixel space
        """
        # Ultralytics predict can accept numpy RGB
        results = self.model.predict(source=img_rgb, conf=conf, imgsz=640, device=self.device, verbose=False)
        if len(results) == 0:
            return []
        res = results[0]
        # res.keypoints exist for pose model; shape (n_person, 17, 3)
        kps_list = []
        try:
            kps = res.keypoints  # Ultralytics KeyPoints object
            # It might be a Tensor or numpy; convert
            arr = kps.xywhk if hasattr(kps, 'xywhk') else None
        except Exception:
            arr = None

        # Fallback: use res.keypoints.data if available
        if arr is None:
            try:
                # res.keypoints.data returns tensor-like
                data = res.keypoints.data.cpu().numpy()
                # data shape (num_persons, 17, 3)
                for person in data:
                    kps_list.append(person.copy())
            except Exception:
                # No keypoints found
                return []
        else:
            # If special attributes exist
            try:
                data = res.keypoints.data.cpu().numpy()
                for person in data:
                    kps_list.append(person.copy())
            except Exception:
                return []

        return kps_list

    def predict_sequence(self, frames_rgb: List[np.ndarray], conf: float = 0.25):
        """
        frames_rgb: list of H,W,3 (RGB) frames
        returns stacked numpy array: (T, P_max, 17, 3) with -1 padding when fewer persons.
        For Smart-Coach we will pick the top-scored person (player) later.
        """
        all_kps = []
        for f in frames_rgb:
            kps = self.predict_frame(f, conf=conf)
            all_kps.append(kps)
        return all_kps  # list per frame; each is list of persons (K,3)
