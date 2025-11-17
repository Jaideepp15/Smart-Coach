import numpy as np

class Diagnoser:
    def __init__(self):
        self.elbow_thresh = 8.0
        self.release_sync_ms = 80

    def diagnose(self, features, ref_features):
        out = []
        le = np.array(features.get('left_elbow',[]))
        re = np.array(features.get('right_elbow',[]))
        rle = np.array(ref_features.get('left_elbow',[]))
        rre = np.array(ref_features.get('right_elbow',[]))
        if le.size and rle.size:
            diff_l = float(np.mean(np.abs(le - rle)))
            if diff_l > self.elbow_thresh:
                out.append({'issue':'left_elbow_misaligned','desc':f'{diff_l:.1f}° deviation'})
        if re.size and rre.size:
            diff_r = float(np.mean(np.abs(re - rre)))
            if diff_r > self.elbow_thresh:
                out.append({'issue':'right_elbow_misaligned','desc':f'{diff_r:.1f}° deviation'})
        # wrist timing heuristic
        wh = np.array(features.get('wrist_height',[]))
        rwh = np.array(ref_features.get('wrist_height',[]))
        if wh.size and rwh.size:
            my_peak = int(np.argmax(wh))
            ref_peak = int(np.argmax(rwh))
            fps = features.get('fps',30)
            dt_ms = abs(my_peak - ref_peak) / fps * 1000
            if dt_ms > self.release_sync_ms:
                out.append({'issue':'release_timing_off','desc':f'{int(dt_ms)} ms timing difference'})
        return out
