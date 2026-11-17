"""MLP surface map and out-of-sample wrapper."""
import numpy as np
import torch
from torch import nn


class SurfaceMLP(nn.Module):
    """d -> 64 -> 64 -> 64 -> 3 with SiLU (smooth, so Jacobian losses are differentiable)."""

    def __init__(self, d, width=64):
        super().__init__()
        self.d, self.width = d, width
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 3),
        )

    def forward(self, x):
        return self.net(x)


class AEPModel:
    """Maps raw rows (applying the stored PCA transform) to R^3."""

    def __init__(self, net, transform):
        self.net = net.eval()
        self.transform = transform

    @torch.no_grad()
    def embed_working(self, X, chunk=8192):
        dtype = next(self.net.parameters()).dtype
        out = [self.net(torch.as_tensor(X[s:s + chunk], dtype=dtype)).numpy()
               for s in range(0, len(X), chunk)]
        return np.concatenate(out).astype(np.float64)

    def __call__(self, X_raw):
        return self.embed_working(self.transform.transform(X_raw))

    def save(self, path):
        """Save network weights and the PCA transform for out-of-sample use."""
        tf = self.transform
        torch.save({"d": self.net.d, "width": self.net.width, "state_dict": self.net.state_dict(),
                    "mean": tf.mean, "components": tf.components,
                    "explained_variance_ratio": tf.explained_variance_ratio}, path)

    @classmethod
    def load(cls, path):
        from aep.preprocess import PCATransform
        ckpt = torch.load(path, weights_only=False)
        net = SurfaceMLP(ckpt["d"], ckpt["width"])
        net.load_state_dict(ckpt["state_dict"])
        tf = PCATransform(ckpt["mean"], ckpt["components"], ckpt["explained_variance_ratio"])
        return cls(net, tf)
