"""Fourier Neural Operator surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import BaseSurrogate, DatasetLike, SurrogatePrediction, _extract_xy
from ..train import TrainConfig, train_torch_regression
from ..utils import optional_import, torch_available


def _require_torch():
    if not torch_available():
        raise ImportError("PyTorch is required for FNO models. Install torch to use this feature.")
    return optional_import("torch", extra="torch")


def _require_torch_nn():
    torch = _require_torch()
    return torch, optional_import("torch.nn", extra="torch"), optional_import("torch.nn.functional", extra="torch")


class SpectralConv1d:  # pragma: no cover - exercised when torch is installed
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        torch, nn, _ = _require_torch_nn()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def __call__(self, x):
        torch = _require_torch()
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device
        )
        out_ft[..., : self.modes] = torch.einsum("bix, iox -> box", x_ft[..., : self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d:  # pragma: no cover - exercised when torch is installed
    def __init__(self, modes: int, width: int, depth: int, activation: str = "gelu"):
        torch, nn, F = _require_torch_nn()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.activation = activation
        self.fc0 = nn.Linear(1, width)
        self.spectral_layers = [SpectralConv1d(width, width, modes) for _ in range(depth)]
        self.w_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)
        self.act = getattr(F, activation)

    def parameters(self):
        torch = _require_torch()
        params = []
        params += list(self.fc0.parameters())
        params += list(self.fc1.parameters())
        params += list(self.fc2.parameters())
        for layer in self.w_layers:
            params += list(layer.parameters())
        for spectral in self.spectral_layers:
            params += list(spectral.weights)
        return params

    def __call__(self, x):
        torch = _require_torch()
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for spectral, w in zip(self.spectral_layers, self.w_layers):
            x1 = spectral(x)
            x2 = w(x)
            x = x1 + x2
            x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


@dataclass
class FNOOperatorSurrogate(BaseSurrogate):
    """Minimal FNO-based operator surrogate."""

    config: Optional[dict] = None
    model_: Optional[Any] = None

    def _build_model(self) -> Any:
        cfg = self.config or {}
        modes = int(cfg.get("modes", 8))
        width = int(cfg.get("width", 32))
        depth = int(cfg.get("depth", 4))
        activation = cfg.get("activation", "gelu")
        return FNO1d(modes=modes, width=width, depth=depth, activation=activation)

    def fit(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "FNOOperatorSurrogate":
        x, y = _extract_xy(dataset, y)
        self.model_ = self._build_model()
        config = TrainConfig.from_kwargs(kwargs)
        train_torch_regression(self.model_, x, y, config=config)
        return self

    def predict(
        self,
        x: np.ndarray,
        params: Optional[np.ndarray] = None,
        return_uncertainty: bool = False,
        **kwargs: Any,
    ) -> SurrogatePrediction:
        if self.model_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        torch = _require_torch()
        self.model_.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)
        with torch.no_grad():
            pred = self.model_(x_tensor).cpu().numpy()
        variance = None
        if return_uncertainty:
            variance = np.var(pred, axis=0, keepdims=True)
        return SurrogatePrediction(mean=pred, variance=variance, metadata={"model": "FNO"})
