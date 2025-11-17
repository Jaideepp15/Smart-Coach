import onnxruntime as ort
import numpy as np

class HybrIKONNX:
    def __init__(self, onnx_path=None, device='cpu'):
        self.has_model = False
        if onnx_path and os.path.exists(onnx_path):
            providers = ['CPUExecutionProvider']
            if device.startswith('cuda'):
                providers = ['CUDAExecutionProvider','CPUExecutionProvider']
            self.sess = ort.InferenceSession(onnx_path, providers=providers)
            self.inp_name = self.sess.get_inputs()[0].name
            self.has_model = True
        else:
            self.sess = None
            self.inp_name = None
            self.has_model = False

    def lift_sequence(self, kps2d):
        """
        kps2d: (T, N, 3) x,y,conf in crop pixel coords
        Returns (T, N, 3) world-ish coords
        """
        T,N,_ = kps2d.shape
        if not self.has_model:
            # fallback heuristic lifter
            xy = kps2d[..., :2].astype(float)
            xy_norm = (xy - xy.mean(axis=0, keepdims=True)) / (xy.std(axis=0, keepdims=True) + 1e-6)
            vel = np.vstack([np.zeros((1,N,2)), np.diff(xy_norm, axis=0)])
            z = np.linalg.norm(vel, axis=-1, keepdims=True)
            out = np.zeros((T,N,3))
            out[..., :2] = xy_norm
            out[..., 2:] = z
            return out
        # prepare input: adapt to model's expected input format
        # many HybrIK ONNXs expect flattened 2D points shape (1, T, N*2) or similar — adjust if needed.
        inp = kps2d[..., :2].astype('float32')
        try:
            outputs = self.sess.run(None, {self.inp_name: inp[None,...]})
            out = outputs[0].squeeze()
            return out
        except Exception:
            # if inference fails, fallback
            xy = kps2d[..., :2].astype(float)
            xy_norm = (xy - xy.mean(axis=0, keepdims=True)) / (xy.std(axis=0, keepdims=True) + 1e-6)
            vel = np.vstack([np.zeros((1,N,2)), np.diff(xy_norm, axis=0)])
            z = np.linalg.norm(vel, axis=-1, keepdims=True)
            out = np.zeros((T,N,3))
            out[..., :2] = xy_norm
            out[..., 2:] = z
            return out
