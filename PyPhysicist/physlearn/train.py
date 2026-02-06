"""Training utilities for PhysLearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .utils import optional_import, torch_available


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "TrainConfig":
        return cls(
            epochs=int(kwargs.get("epochs", 100)),
            lr=float(kwargs.get("lr", 1e-3)),
            batch_size=int(kwargs.get("batch_size", 32)),
            weight_decay=float(kwargs.get("weight_decay", 0.0)),
            device=str(kwargs.get("device", "cpu")),
            seed=int(kwargs.get("seed", 42)),
        )


def _require_torch():
    if not torch_available():
        raise ImportError("PyTorch is required for training utilities. Install torch to use this feature.")
    return optional_import("torch", extra="torch")


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int):
    torch = _require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_torch_regression(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    callbacks: Optional[Iterable[Callable[..., Any]]] = None,
    deeponet_split: Optional[int] = None,
) -> None:
    torch = _require_torch()
    nn = optional_import("torch.nn", extra="torch")
    optim = optional_import("torch.optim", extra="torch")
    torch.manual_seed(config.seed)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    loader = _make_loader(x, y, config.batch_size)
    for epoch in range(config.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            if deeponet_split is not None:
                pred = model(batch_x[:, :deeponet_split], batch_x[:, deeponet_split:])
            else:
                pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
        if callbacks:
            for cb in callbacks:
                cb(epoch=epoch, loss=float(loss.detach().cpu().item()))


def train_pinn(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    residual_fn: Optional[Callable[..., Any]],
    config: TrainConfig,
    adaptive_loss: bool = False,
    w_data: float = 1.0,
    w_res: float = 1.0,
    w_bc: float = 1.0,
    w_reg: float = 0.0,
    callbacks: Optional[Iterable[Callable[..., Any]]] = None,
    **kwargs: Any,
) -> None:
    torch = _require_torch()
    nn = optional_import("torch.nn", extra="torch")
    optim = optional_import("torch.optim", extra="torch")
    torch.manual_seed(config.seed)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    loader = _make_loader(x, y, config.batch_size)
    weights = np.array([w_data, w_res, w_bc, w_reg], dtype=float)
    for epoch in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.requires_grad_(True)
            pred = model(batch_x)
            loss_data = loss_fn(pred, batch_y)
            loss_res = torch.tensor(0.0, device=batch_x.device)
            if residual_fn is not None:
                loss_res = residual_fn(model, batch_x, pred)
            loss_bc = torch.tensor(0.0, device=batch_x.device)
            loss_reg = torch.tensor(0.0, device=batch_x.device)
            loss = weights[0] * loss_data + weights[1] * loss_res + weights[2] * loss_bc + weights[3] * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if adaptive_loss:
                weights = weights / np.clip(weights.sum(), 1e-12, None)
        if callbacks:
            for cb in callbacks:
                cb(epoch=epoch, loss=float(loss.detach().cpu().item()))
