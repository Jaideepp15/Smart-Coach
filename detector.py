from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        self.model = YOLO(model_path)
        # ultralytics auto-selects device; you can override via environment var if needed
        self.device = device

    def detect(self, image, conf=0.35, classes=[0]):
        """
        image: RGB numpy array (H,W,3)
        classes: [0] for person in COCO
        returns list of boxes: (x1,y1,x2,y2,score)
        """
        results = self.model.predict(source=image, imgsz=640, conf=conf, classes=classes, verbose=False)
        if len(results) == 0:
            return []
        r = results[0]
        boxes = []
        if r.boxes is None:
            return []
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), score))
        return boxes
