# detector.py
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='models/yolov11n.pt', device='cpu'):
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
        except Exception:
            self.model = None

    def detect(self, image_rgb, conf=0.35):
        """
        returns list of bboxes (x1,y1,x2,y2,score)
        """
        if self.model is None:
            h,w = image_rgb.shape[:2]
            return [(int(0.05*w), int(0.05*h), int(0.95*w), int(0.95*h), 0.6)]
        res = self.model.predict(source=image_rgb, imgsz=640, conf=conf, device='cpu', verbose=False)
        if len(res) == 0 or res[0].boxes is None:
            return []
        boxes=[]
        for b in res[0].boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            score = float(b.conf[0])
            boxes.append((x1,y1,x2,y2,score))
        return boxes
