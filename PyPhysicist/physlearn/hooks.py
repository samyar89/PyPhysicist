"""Training hooks and callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class History:
    losses: List[float] = field(default_factory=list)

    def __call__(self, *, loss: float, **kwargs: Any) -> None:
        self.losses.append(loss)


@dataclass
class EarlyStopping:
    patience: int = 10
    best_loss: Optional[float] = None
    wait: int = 0
    stopped_epoch: Optional[int] = None

    def __call__(self, *, epoch: int, loss: float, **kwargs: Any) -> None:
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
