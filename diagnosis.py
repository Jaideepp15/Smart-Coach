import numpy as np

class Diagnoser:
    def __init__(self):
        self.elbow_thresh = 8.0
        self.release_sync_ms = 80

    def diagnose(self, features, ref_features):
        out=[]
        le=np.array(features.get('left_elbow',[])); rle=np.array(ref_features.get('left_elbow',[]))
        if le.size and rle.size:
            if float(np.mean(np.abs(le-rle)))>self.elbow_thresh:
                out.append({'issue':'left_elbow_misaligned'})
        wh=np.array(features.get('wrist_height',[])); rwh=np.array(ref_features.get('wrist_height',[]))
        if wh.size and rwh.size:
            my_peak=int(np.argmax(wh)); ref_peak=int(np.argmax(rwh))
            fps=features.get('fps',30); dt_ms=abs(my_peak-ref_peak)/fps*1000
            if dt_ms>self.release_sync_ms:
                out.append({'issue':'release_timing_off','desc':f'{int(dt_ms)} ms'})
        return out
