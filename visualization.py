import cv2
import numpy as np
import tempfile

SKELETON_EDGES = [(11,13),(13,15),(12,14),(14,16),(11,12)]

class Visualizer:
    def draw_skeleton(self, img, kps, color=(255,0,0)):
        for (x,y,c) in kps:
            cv2.circle(img, (int(x),int(y)), 3, color, -1)
        for a,b in SKELETON_EDGES:
            if a < kps.shape[0] and b < kps.shape[0]:
                x1,y1,_ = kps[a]
                x2,y2,_ = kps[b]
                cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        return img

    def create_overlay(self, frames, keypoints2d, ref_clip=None):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h,w = frames[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(tmp.name, fourcc, 20.0, (w,h))
        for i,f in enumerate(frames):
            img = f.copy()
            if i < len(keypoints2d):
                kps = keypoints2d[i]
                img = self.draw_skeleton(img, kps, color=(255,0,0))
            # reference ghost
            if ref_clip and isinstance(ref_clip.get('kps2d',None), list):
                try:
                    rk = np.array(ref_clip['kps2d'][min(i, len(ref_clip['kps2d'])-1)])
                    img = self.draw_skeleton(img, rk, color=(0,200,0))
                except Exception:
                    pass
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        return tmp.name
