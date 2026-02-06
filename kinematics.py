"""
Angle-Based Basketball Shot Biomechanics
Camera-invariant, physics-free, KG-ready.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import savgol_filter


# =========================
# Utilities
# =========================

def safe_savgol(arr, window, poly, deriv=0, delta=1.0):
    n = len(arr)
    if n < 3:
        return arr.copy() if deriv == 0 else np.zeros_like(arr)
    w = window if window % 2 == 1 else window - 1
    w = max(3, min(w, n if n % 2 else n - 1))
    poly = min(poly, w - 1)
    return savgol_filter(arr, w, poly, deriv=deriv, delta=delta, mode="interp")


def angle_3d(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# =========================
# Main Class
# =========================

class Biomechanics:
    """
    Angle-only biomechanics extraction.
    Designed for shooter similarity + KG reasoning.
    """

    def __init__(self):
        self.idx = {
            "R_SH": 6, "R_EL": 8, "R_WR": 10,
            "L_SH": 5, "L_EL": 7, "L_WR": 9,
            "R_HIP": 12, "L_HIP": 11,
            "R_KNEE": 14, "L_KNEE": 13,
            "R_ANK": 16, "L_ANK": 15
        }

    # -------------------------
    # Angle sequences
    # -------------------------

    def angle_sequence(self, seq, a, b, c):
        return np.array([
            angle_3d(seq[t, self.idx[a]],
                     seq[t, self.idx[b]],
                     seq[t, self.idx[c]])
            for t in range(seq.shape[0])
        ])

    # -------------------------
    # Phase detection
    # -------------------------

    def detect_phases(self, seq, fps):
        knee = self.angle_sequence(seq, "R_HIP", "R_KNEE", "R_ANK")
        knee_s = safe_savgol(knee, 11, 2)
        
        # Find all local minima (potential dips)
        from scipy.signal import find_peaks
        dip_candidates, _ = find_peaks(-knee_s)  # Negative for minima
        
        if len(dip_candidates) == 0:
            # No clear dip - might be catch-and-shoot
            # Use elbow as proxy: when does elbow start extending rapidly?
            elbow = self.angle_sequence(seq, "R_SH", "R_EL", "R_WR")
            elbow_v = np.gradient(elbow)
            dip = int(np.argmax(elbow_v > np.percentile(elbow_v, 90)))
        else:
            # Pick the deepest dip in first 60% of sequence
            valid_dips = dip_candidates[dip_candidates < int(0.6 * len(knee_s))]
            if len(valid_dips) == 0:
                dip = dip_candidates[0]
            else:
                dip = valid_dips[np.argmin(knee_s[valid_dips])]
        
        # Validate dip makes sense
        if knee_s[dip] > 140:  # Not flexed enough to be a real dip
            # Probably catch-and-shoot, use first 20% of sequence
            dip = int(0.2 * len(knee_s))

        extend = dip + int(np.argmax(knee_s[dip:]))

        release = min(len(knee_s) - 1, extend + int(0.1 * fps))

        return {
            "dip": dip,
            "extension": extend,
            "release": release
        }

    # -------------------------
    # Main extraction
    # -------------------------

    def extract_biomechanics(
        self,
        seq3d: np.ndarray,
        fps: float,
        shooting_hand: str = "right",
        manual_release: Optional[int] = None
    ) -> Dict[str, Any]:

        T = seq3d.shape[0]
        dt = 1.0 / fps

        # Side selection
        if shooting_hand == "left":
            SH, EL, WR = "L_SH", "L_EL", "L_WR"
            HIP, KNEE, ANK = "L_HIP", "L_KNEE", "L_ANK"
        else:
            SH, EL, WR = "R_SH", "R_EL", "R_WR"
            HIP, KNEE, ANK = "R_HIP", "R_KNEE", "R_ANK"

        phases = self.detect_phases(seq3d, fps)
        release = manual_release if manual_release is not None else phases["release"]

        # Angles
        elbow = self.angle_sequence(seq3d, SH, EL, WR)
        shoulder = self.angle_sequence(seq3d, EL, SH, HIP)
        hip = self.angle_sequence(seq3d, SH, HIP, KNEE)
        knee = self.angle_sequence(seq3d, HIP, KNEE, ANK)

        # Smooth
        elbow_s = safe_savgol(elbow, 11, 2)
        knee_s = safe_savgol(knee, 11, 2)

        # Angular velocities
        elbow_v = np.gradient(elbow_s, dt)
        knee_v = np.gradient(knee_s, dt)

        # Drive window
        d0, d1 = phases["dip"], phases["extension"]

        return {
            # Frames
            "release_frame": int(release),
            "dip_frame": int(phases["dip"]),
            "extension_frame": int(phases["extension"]),

            # Angles
            "angles_at_release": {
                "elbow": float(elbow_s[release]),
                "shoulder": float(shoulder[release]),
                "hip": float(hip[release]),
                "knee": float(knee_s[release])
            },

            "angles_at_dip": {
                "elbow": float(elbow_s[phases["dip"]]),
                "knee": float(knee_s[phases["dip"]])
            },

            # ROM
            "range_of_motion": {
                "elbow_extension": float(np.max(elbow_s[d0:d1+1]) - np.min(elbow_s[d0:d1+1])),
                "knee_extension": float(np.max(knee_s[d0:d1+1]) - np.min(knee_s[d0:d1+1]))
            },

            # Timing
            "timing": {
                "dip_duration_s": float((phases["dip"]) / fps),
                "drive_duration_s": float((d1 - d0) / fps)
            },

            # Coordination
            "coordination": {
                "knee_to_elbow_delay_s": float(
                    (np.argmax(elbow_v[d0:d1+1]) - np.argmax(knee_v[d0:d1+1])) / fps
                )
            },

            # Smoothness
            "consistency": {
                "elbow_smoothness": float(np.std(np.gradient(elbow_v[d0:d1+1], dt))),
                "knee_smoothness": float(np.std(np.gradient(knee_v[d0:d1+1], dt)))
            }
        }
    
    #For validation
    def compute_angle_timeseries(self, kps3d, shooting_hand="Right-handed"):
        """
        Returns per-frame joint angle trajectories for validation.
        """

        def angle(a, b, c):
            ba = a - b
            bc = c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
            cosang = np.dot(ba, bc) / denom
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        right_elbow, left_elbow = [], []
        right_knee, left_knee = [], []

        for f in kps3d:
            # COCO indices
            right_elbow.append(angle(f[6], f[8], f[10]))
            left_elbow.append(angle(f[5], f[7], f[9]))
            right_knee.append(angle(f[12], f[14], f[16]))
            left_knee.append(angle(f[11], f[13], f[15]))

        return {
            "right_elbow": np.array(right_elbow),
            "left_elbow": np.array(left_elbow),
            "right_knee": np.array(right_knee),
            "left_knee": np.array(left_knee),
        }

