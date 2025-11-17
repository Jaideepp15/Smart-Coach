import numpy as np

class STCFormerSmoother:
    def __init__(self, kernel_size=5):
        self.k = kernel_size

    def smooth(self, seq):
        T,N,C = seq.shape
        out = np.copy(seq)
        half = self.k // 2
        for t in range(T):
            lo = max(0, t-half)
            hi = min(T, t+half+1)
            out[t] = seq[lo:hi].mean(axis=0)
        return out
