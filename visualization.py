# visualization.py
import cv2
import numpy as np
import os

class Visualizer:
    def __init__(self):
        pass

    def draw_skeleton(self, frame_rgb, keypoints, color=(0,255,0)):
        # keypoints: (17,3) x,y,conf
        pairs = [
            (5,7),(7,9),(6,8),(8,10),  # arms
            (11,13),(13,15),(12,14),(14,16), # legs
            (5,6),(11,12),(5,11),(6,12),(1,2)
        ]
        frame = frame_rgb.copy()
        H,W = frame.shape[:2]
        for a,b in pairs:
            xa,ya,ca = keypoints[a]
            xb,yb,cb = keypoints[b]
            if ca>0.1 and cb>0.1:
                cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), color, 2)
        for i,(x,y,c) in enumerate(keypoints):
            if c>0.1:
                cv2.circle(frame, (int(x),int(y)), 3, (0,0,255), -1)
        return frame
    
    def draw_phase_marker(self, frame, keypoints, label, color):
        """
        Draws a labeled marker near the shooting wrist.
        """
        wrist = keypoints[10]  # Right wrist (COCO index)
        x, y, conf = wrist
        if conf < 0.1:
            return frame

        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.putText(frame, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA)
        return frame

    def create_overlay(self, frames, kps2d, output_path="output_overlay.mp4", fps=30, dip_frame=None, extension_frame=None, release_frame=None):
        """
        frames: list of RGB frames (H,W,3)
        kps2d: (T,17,3) in pixel coords
        """

        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        for i, frame in enumerate(frames):
            kp = kps2d[i]
            f = self.draw_skeleton(frame, kp)

            # ---- Phase Annotations ----
            if dip_frame is not None and i == dip_frame:
                f = self.draw_phase_marker(f, kp, "DIP", (255, 0, 0))

            if extension_frame is not None and i == extension_frame:
                f = self.draw_phase_marker(f, kp, "EXTENSION", (0, 165, 255))

            if release_frame is not None and i == release_frame:
                f = self.draw_phase_marker(f, kp, "RELEASE", (0, 0, 255))

            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        return output_path
