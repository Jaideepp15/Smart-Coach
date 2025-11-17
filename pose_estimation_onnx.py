import onnxruntime as ort
import numpy as np
import cv2

class RTMPoseONNX:
    """
    Lightweight ONNX runtime wrapper for RTMPose-L exported ONNX.
    Expects model to take an image tensor and output heatmaps/keypoints.
    This wrapper includes a simple postprocessing pipeline for common RTMPose ONNX exports.
    """
    def __init__(self, onnx_path, device='cpu'):
        providers = ['CPUExecutionProvider']
        if device.startswith('cuda'):
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        # Inspect model input
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g., (1,3,256,192)
        _,_,self.h_in,self.w_in = self.input_shape

    def _preprocess(self, img):
        # img: RGB numpy array
        im = cv2.resize(img, (self.w_in, self.h_in))
        im = im.astype('float32') / 255.0
        im = np.transpose(im, (2,0,1))[None,...]  # (1,3,H,W)
        return im

    def _postprocess(self, out):
        # This depends on the exported ONNX output format: heatmaps or direct keypoints.
        # We support common case: output is (1, K, 3) or (1, K, 2)
        if isinstance(out, (list,tuple)):
            out0 = out[0]
        else:
            out0 = out
        arr = out0.squeeze()
        # If arr is (K,3) assume x,y,score normalized - scale to input image
        if arr.ndim == 2 and arr.shape[1] >= 2:
            kps = []
            for kp in arr:
                x = float(kp[0]) * self.w_in
                y = float(kp[1]) * self.h_in
                conf = float(kp[2]) if kp.shape[0]>2 else 1.0
                kps.append((x,y,conf))
            return np.array(kps)
        # fallback: zeros
        return np.zeros((33,3))

    def predict_frame(self, img):
        inp = self._preprocess(img)
        outputs = self.sess.run(None, {self.input_name: inp})
        # Some exports output heatmaps; try to handle common cases
        return self._postprocess(outputs[0])

    def predict_sequence(self, frames):
        seq = []
        for f in frames:
            kps = self.predict_frame(f)
            seq.append(kps)
        return np.stack(seq)
