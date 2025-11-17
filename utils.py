import cv2
import numpy as np
import tempfile
from pathlib import Path
import torch
from loguru import logger

def load_video_frames(file_like, max_frames=900):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(file_like.read())
    tmp.flush()
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()
    return frames, int(fps)

class PersonDetector:
    """
    Light wrapper around RTMDet (mmdet) or YOLOv10. Provide config/checkpoint paths via args.
    """
    def __init__(self, model_cfg=None, ckpt=None, device='cpu'):
        self.device = device
        self.model_cfg = model_cfg
        self.ckpt = ckpt
        # Lazy import to reduce startup time
        try:
            from mmdet.apis import init_detector, inference_detector
            self.init_detector = init_detector
            self.inference_detector = inference_detector
            if model_cfg and ckpt:
                self.det_model = init_detector(model_cfg, ckpt, device=device)
            else:
                self.det_model = None
        except Exception as e:
            logger.warning('mmdet not available: %s', e)
            self.det_model = None

    def detect(self, frame, score_thr=0.4):
        if self.det_model is None:
            # fallback: center box
            h,w = frame.shape[:2]
            return [(int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95), 1.0)]
        dets = self.inference_detector(self.det_model, frame)
        # mmdet returns list per class; person often class id 0
        person_results = dets[0] if isinstance(dets, (list,tuple)) else dets
        boxes = []
        for box in person_results:
            x1,y1,x2,y2,score = box.tolist()
            if score>=score_thr:
                boxes.append((int(x1),int(y1),int(x2),int(y2),float(score)))
        if not boxes:
            h,w = frame.shape[:2]
            return [(int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95), 0.5)]
        return boxes

def smart_crop_sequence(frames, detector, pad=1.2, size=512):
    """
    Detect per-frame boxes, smooth them temporally, crop & resize to `size`.
    Returns crops list and boxes list (per frame).
    """
    h0,w0 = frames[0].shape[:2]
    boxes = []
    for f in frames:
        b = detector.detect(f)
        boxes.append(b[0])  # take top person
    # temporal smoothing: simple moving average of centers and sizes
    centers = []
    sizes = []
    for (x1,y1,x2,y2,score) in boxes:
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        side = max(x2-x1, y2-y1) * pad
        centers.append((cx,cy))
        sizes.append(side)
    # smooth
    centers = np.array(centers)
    sizes = np.array(sizes)
    kernel = 5
    pad_k = kernel//2
    def smooth_arr(a):
        return np.convolve(a, np.ones(kernel)/kernel, mode='same')
    cx_s = smooth_arr(centers[:,0])
    cy_s = smooth_arr(centers[:,1])
    sz_s = smooth_arr(sizes)

    crops = []
    out_boxes = []
    for i, f in enumerate(frames):
        cx,cy,side = cx_s[i], cy_s[i], sz_s[i]
        x0 = int(max(0, cx - side/2))
        y0 = int(max(0, cy - side/2))
        x1 = int(min(w0, cx + side/2))
        y1 = int(min(h0, cy + side/2))
        crop = f[y0:y1, x0:x1]
        if crop.size==0:
            crop = f
            x0,y0,x1,y1 = 0,0,w0,h0
        crop = cv2.resize(crop, (size,size))
        crops.append(crop)
        out_boxes.append((x0,y0,x1,y1))
    return crops, out_boxes
