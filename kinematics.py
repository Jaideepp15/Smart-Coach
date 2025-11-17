import numpy as np

class Kinematics:
    def __init__(self):
        pass

    @staticmethod
    def angle_between(a,b,c, eps=1e-8):
        ba = a - b
        bc = c - b
        num = np.sum(ba*bc, axis=-1)
        den = (np.linalg.norm(ba,axis=-1)*np.linalg.norm(bc,axis=-1) + eps)
        cosang = np.clip(num/den, -1, 1)
        return np.degrees(np.arccos(cosang))

    def extract_features(self, keypoints3d, fps=30):
        """
        keypoints3d: (T, N, 3) array. Indices follow COCO/Mediapipe mapping.
        Returns dict of timeseries features useful for basketball shot analysis.
        """
        T, N, _ = keypoints3d.shape
        features = {'fps': fps}
        # Indices (COCO-style) assumption — adapt for your 2D keypoint set mapping.
        # Using common convention: 11=left_shoulder,13=left_elbow,15=left_wrist, 12=right_shoulder,14=right_elbow,16=right_wrist
        try:
            L_sh = keypoints3d[:,11]
            L_el = keypoints3d[:,13]
            L_wr = keypoints3d[:,15]
            R_sh = keypoints3d[:,12]
            R_el = keypoints3d[:,14]
            R_wr = keypoints3d[:,16]
        except Exception:
            # If mapping doesn't match, return zeros
            features.update({
                'left_elbow': [0]*T, 'right_elbow':[0]*T,
                'wrist_height':[0]*T, 'hip_knee_sync':[0]*T
            })
            return features

        left_elbow_ang = self.angle_between(L_sh, L_el, L_wr)
        right_elbow_ang = self.angle_between(R_sh, R_el, R_wr)
        # wrist height (z) — use z coord if present
        try:
            wrist_z = (L_wr[:,2] + R_wr[:,2]) / 2.0
        except Exception:
            wrist_z = np.zeros(T)

        # knee-hip sync: use hip and knee indices if available (e.g., 23/24 hips, 25/26 knees in some sets)
        hip_idx = 23 if N>23 else None
        knee_idx = 25 if N>25 else None
        if hip_idx and knee_idx:
            hip = keypoints3d[:,hip_idx]
            knee = keypoints3d[:,knee_idx]
            hip_knee_sync = np.linalg.norm(np.diff(hip, axis=0), axis=-1)
            hip_knee_sync = np.concatenate([[0], hip_knee_sync])
        else:
            hip_knee_sync = np.zeros(T)

        features['left_elbow'] = left_elbow_ang.tolist()
        features['right_elbow'] = right_elbow_ang.tolist()
        features['wrist_height'] = wrist_z.tolist()
        features['hip_knee_sync'] = hip_knee_sync.tolist()
        return features
