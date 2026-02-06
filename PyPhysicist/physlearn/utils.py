"""Utility helpers for PhysLearn."""

from __future__ import annotations

from typing import Optional


def optional_import(module: str, extra: Optional[str] = None):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised via explicit checks
        hint = f" Install the optional dependency '{extra}'." if extra else ""
        raise ImportError(f"Optional dependency '{module}' is required.{hint}") from exc


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
