"""End-to-end training and evaluation pipeline for PhysLearn."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from PyPhysicist.physlearn import (
    PhysLearn,
    burgers_1d,
    coverage_probability,
    crps_gaussian,
    nll_gaussian,
    nrmse,
    rmse,
)
from PyPhysicist.physlearn.data_assimilation import EnKF
from PyPhysicist.physlearn.hooks import History
from PyPhysicist.physlearn.utils import torch_available

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "artifacts_dir": "artifacts",
    "data": {
        "n_train_low": 800,
        "n_train_high": 80,
        "n_val": 80,
        "n_test": 100,
        "viscosity_low": 0.02,
        "viscosity_high": 0.01,
    },
    "models": {
        "fno": {"modes": 16, "width": 64, "depth": 4, "activation": "gelu"},
        "pinn": {"width": 128, "depth": 4},
        "hybrid": {"width": 64, "depth": 3},
    },
    "training": {
        "fno": {"epochs": 200, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-6},
        "pinn": {"epochs": 600, "batch_size": 32, "lr": 1e-3, "adaptive_loss": True},
        "hybrid": {
            "pretrain_epochs": 150,
            "finetune_epochs": 150,
            "batch_size": 32,
            "lr": 1e-3,
        },
    },
    "evaluation": {
        "coverage_alpha": 0.95,
        "calibration_bins": 9,
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised when yaml missing
        raise ImportError("PyYAML is required to load configuration files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch_available():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    np.savez(path, **data)


def _calibration_curve(mean: np.ndarray, std: np.ndarray, target: np.ndarray, bins: int) -> Dict[str, Any]:
    alphas = np.linspace(0.1, 0.9, bins)
    coverages = [coverage_probability(mean, std, target, alpha=float(alpha)) for alpha in alphas]
    return {"alphas": alphas.tolist(), "coverages": coverages}


def _save_calibration_plot(path: Path, curve: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised when matplotlib missing
        raise ImportError("matplotlib is required to save calibration plots.") from exc
    alphas = np.array(curve["alphas"])
    coverages = np.array(curve["coverages"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(alphas, coverages, marker="o", label="Empirical")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _split_burgers(
    n_train_low: int,
    n_train_high: int,
    n_val: int,
    n_test: int,
    viscosity_low: float,
    viscosity_high: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_low = burgers_1d(n_samples=n_train_low, viscosity=viscosity_low, seed=seed)
    train_high = burgers_1d(n_samples=n_train_high, viscosity=viscosity_high, seed=seed + 1)
    val = burgers_1d(n_samples=n_val, viscosity=viscosity_high, seed=seed + 2)
    test = burgers_1d(n_samples=n_test, viscosity=viscosity_high, seed=seed + 3)
    return train_low, train_high, val, test


def _estimate_uncertainty(
    mean: np.ndarray, std: np.ndarray | None, val: Dict[str, np.ndarray], pred_val: np.ndarray
) -> np.ndarray:
    if std is not None and np.all(std > 0):
        return std
    residual_std = float(np.std(val["y"] - pred_val))
    if residual_std == 0:
        residual_std = 1e-6
    return np.full_like(mean, residual_std)


def _evaluate_model(
    model_name: str,
    surrogate: Any,
    val: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    coverage_alpha: float,
    calibration_bins: int,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    pred_val = surrogate.predict(val["x"]).mean
    prediction = surrogate.predict(test["x"], return_uncertainty=True)
    mean = prediction.mean
    std = _estimate_uncertainty(mean, prediction.std, val, pred_val)
    metrics = {
        "rmse": rmse(mean, test["y"]),
        "nrmse": nrmse(mean, test["y"]),
        "nll": nll_gaussian(mean, std, test["y"]),
        "crps": crps_gaussian(mean, std, test["y"]),
        "coverage": coverage_probability(mean, std, test["y"], alpha=coverage_alpha),
    }
    metrics_path = artifacts_dir / "metrics" / f"{model_name}_metrics.json"
    _save_json(metrics_path, metrics)
    curve = _calibration_curve(mean, std, test["y"], bins=calibration_bins)
    _save_json(artifacts_dir / "metrics" / f"{model_name}_calibration.json", curve)
    _save_calibration_plot(artifacts_dir / "plots" / f"{model_name}_calibration.png", curve)
    return metrics


def _coarse_solver_factory(viscosity: float):
    def _coarse_solver(x: np.ndarray, _params: Any) -> np.ndarray:
        return burgers_1d(n_samples=x.shape[0], viscosity=viscosity, seed=0)["y"]

    return _coarse_solver


def train_and_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config["seed"])
    device = _resolve_device(str(config["device"]))
    _seed_everything(seed)

    artifacts_dir = Path(config["artifacts_dir"])
    dirs = {
        "models": artifacts_dir / "models",
        "metrics": artifacts_dir / "metrics",
        "plots": artifacts_dir / "plots",
        "configs": artifacts_dir / "configs",
        "datasets": artifacts_dir / "datasets",
        "logs": artifacts_dir / "logs",
    }
    for path in dirs.values():
        _ensure_dir(path)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_json(dirs["configs"] / f"run_{timestamp}.json", config)

    data_cfg = config["data"]
    train_low, train_high, val, test = _split_burgers(
        n_train_low=int(data_cfg["n_train_low"]),
        n_train_high=int(data_cfg["n_train_high"]),
        n_val=int(data_cfg["n_val"]),
        n_test=int(data_cfg["n_test"]),
        viscosity_low=float(data_cfg["viscosity_low"]),
        viscosity_high=float(data_cfg["viscosity_high"]),
        seed=seed,
    )

    _save_npz(dirs["datasets"] / "burgers_train_low.npz", train_low)
    _save_npz(dirs["datasets"] / "burgers_train_high.npz", train_high)
    _save_npz(dirs["datasets"] / "burgers_val.npz", val)
    _save_npz(dirs["datasets"] / "burgers_test.npz", test)

    pl = PhysLearn(equation="burgers_1d", device=device, seed=seed)

    results: Dict[str, Any] = {}

    fno_history = History()
    pl.build_surrogate("fno", config=config["models"]["fno"])
    pl.train(train_high, device=device, seed=seed, callbacks=[fno_history], **config["training"]["fno"])
    fno_model = pl.surrogate
    if fno_model is None:
        raise RuntimeError("FNO training did not produce a surrogate model.")
    fno_model.save(str(dirs["models"] / "fno_main.npz"))
    _save_json(dirs["logs"] / "fno_main_losses.json", {"losses": fno_history.losses})
    results["fno_main"] = _evaluate_model(
        "fno_main",
        fno_model,
        val,
        test,
        coverage_alpha=float(config["evaluation"]["coverage_alpha"]),
        calibration_bins=int(config["evaluation"]["calibration_bins"]),
        artifacts_dir=artifacts_dir,
    )

    pinn_history = History()
    pl.build_surrogate("pinn", config=config["models"]["pinn"])
    pl.train(train_high, device=device, seed=seed, callbacks=[pinn_history], **config["training"]["pinn"])
    pinn_model = pl.surrogate
    if pinn_model is None:
        raise RuntimeError("PINN training did not produce a surrogate model.")
    pinn_model.save(str(dirs["models"] / "pinn_baseline.npz"))
    _save_json(dirs["logs"] / "pinn_baseline_losses.json", {"losses": pinn_history.losses})
    results["pinn_baseline"] = _evaluate_model(
        "pinn_baseline",
        pinn_model,
        val,
        test,
        coverage_alpha=float(config["evaluation"]["coverage_alpha"]),
        calibration_bins=int(config["evaluation"]["calibration_bins"]),
        artifacts_dir=artifacts_dir,
    )

    hybrid_history_pre = History()
    hybrid_history_fine = History()
    coarse_solver = _coarse_solver_factory(viscosity=float(data_cfg["viscosity_low"]))
    pl.build_surrogate("hybrid", coarse_solver=coarse_solver, config=config["models"]["hybrid"])
    mf_trainer = pl.build_multifidelity()
    mf_trainer.pretrain(
        train_low,
        device=device,
        seed=seed,
        callbacks=[hybrid_history_pre],
        epochs=int(config["training"]["hybrid"]["pretrain_epochs"]),
        batch_size=int(config["training"]["hybrid"]["batch_size"]),
        lr=float(config["training"]["hybrid"]["lr"]),
    )
    mf_trainer.finetune(
        train_high,
        device=device,
        seed=seed,
        callbacks=[hybrid_history_fine],
        epochs=int(config["training"]["hybrid"]["finetune_epochs"]),
        batch_size=int(config["training"]["hybrid"]["batch_size"]),
        lr=float(config["training"]["hybrid"]["lr"]),
    )
    hybrid_model = pl.surrogate
    if hybrid_model is None:
        raise RuntimeError("Hybrid training did not produce a surrogate model.")
    hybrid_model.save(str(dirs["models"] / "hybrid.npz"))
    _save_json(
        dirs["logs"] / "hybrid_losses.json",
        {"pretrain": hybrid_history_pre.losses, "finetune": hybrid_history_fine.losses},
    )
    results["hybrid"] = _evaluate_model(
        "hybrid",
        hybrid_model,
        val,
        test,
        coverage_alpha=float(config["evaluation"]["coverage_alpha"]),
        calibration_bins=int(config["evaluation"]["calibration_bins"]),
        artifacts_dir=artifacts_dir,
    )

    candidates = np.linspace(0, 1, 50)[:, None]
    acquisition = pl.suggest_experiment("expected_improvement", candidates, budget=10)
    _save_json(
        dirs["metrics"] / "active_learning.json",
        {"selected_indices": acquisition.selected_indices.tolist(), "scores": acquisition.scores.tolist()},
    )

    test_state = test["y"][0].reshape(-1)
    noise = np.random.normal(scale=0.05, size=test_state.shape)
    observations = test_state + noise
    enkf = EnKF()
    forecast_ensemble = observations + np.random.normal(scale=0.1, size=(40, observations.shape[0]))
    forecast_mean = forecast_ensemble.mean(axis=0)
    analysis = enkf.update(forecast_ensemble, observations, observation_noise=1e-3, inflation=1.02)
    err_before = rmse(forecast_mean, test_state)
    err_after = rmse(analysis.analysis_mean, test_state)
    _save_json(
        dirs["metrics"] / "enkf.json",
        {"rmse_before": err_before, "rmse_after": err_after, "improved": err_after <= err_before},
    )

    summary_lines = [
        "# PhysLearn Burgers-1D Results",
        "",
        f"Device: `{device}`",
        "",
        "| Model | RMSE | NRMSE | NLL | CRPS | Coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, metric in results.items():
        summary_lines.append(
            f"| {model_name} | {metric['rmse']:.4e} | {metric['nrmse']:.4e} | "
            f"{metric['nll']:.4e} | {metric['crps']:.4e} | {metric['coverage']:.3f} |"
        )
    summary_lines.append("")
    best_model = min(results, key=lambda name: results[name]["rmse"])
    summary_lines.append(f"Best RMSE model: **{best_model}**.")
    (artifacts_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate PhysLearn models.")
    parser.add_argument("--config", type=Path, required=False, help="Path to YAML config file.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.config is not None:
        config = _merge_dict(config, _load_yaml(args.config))

    if not torch_available():
        raise RuntimeError("PyTorch is required for this training pipeline. Install torch first.")

    train_and_evaluate(config)


if __name__ == "__main__":
    main()
