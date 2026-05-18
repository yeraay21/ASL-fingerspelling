"""Unified entry point.

Usage:
    python main.py --model {gabor_svm|cnn_scratch|mobilenetv2}

Each run creates runs/{model}_{timestamp}/ with checkpoints, logs and scores.
Gabor+SVM follows a different path (sklearn) but writes the same scores schema.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from params import parse_args


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_run_dir(runs_dir: str, model: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = Path(runs_dir) / f"{model}_{stamp}"
    for sub in ("models", "outputs", "scores", "logs", "parameters"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    return run


def _save_params(run_dir: Path, cfg) -> None:
    def _as_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: _as_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, list):
            return [_as_dict(x) for x in obj]
        return obj
    with open(run_dir / "parameters" / "config.json", "w", encoding="utf-8") as f:
        json.dump(_as_dict(cfg), f, indent=2)


def _save_scores(run_dir: Path, scores: dict) -> None:
    with open(run_dir / "scores" / "results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def run_dummy(cfg, run_dir: Path) -> dict:
    """Smoke test: builds loaders and runs one mini-batch through a trivial net."""
    from dataset.loaders import get_loaders

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=cfg.data_dir,
        subjects_train=cfg.subjects_train,
        subjects_test=cfg.subjects_test,
        include_extra=cfg.include_extra,
        img_size=cfg.img_size_small,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        mobilenet_norm=False,
    )
    print(f"[dummy] train batches: {len(train_loader)} | "
          f"val batches: {len(val_loader)} | test batches: {len(test_loader)}")
    xb, yb = next(iter(train_loader))
    print(f"[dummy] batch shape: {tuple(xb.shape)}  labels: {tuple(yb.shape)}")
    return {"model": "dummy", "ok": True,
            "num_classes": len(classes), "classes": classes,
            "train_batches": len(train_loader)}


def run_gabor_svm(cfg, run_dir: Path) -> dict:
    raise NotImplementedError(
        "Gabor+SVM lo implementa Toni. Pipeline esperado:\n"
        "  1. cargar imágenes 64x64 grayscale con dataset.loaders\n"
        "  2. extraer features con dataset.preprocessing.filters.build_gabor_bank\n"
        "  3. PCA(200) -> GridSearchCV(SVC(kernel='rbf'), C=[0.1,1,10])\n"
        "  4. guardar accuracy/precision/recall/f1 en run_dir/scores/results.json"
    )


def run_cnn_scratch(cfg, run_dir: Path, device: torch.device) -> dict:
    from dataset.loaders import get_loaders
    from utility.get_fresh_model import get_fresh_model
    from phases.train import train_one_ep
    from phases.infer import infer_one_ep
    from utility.save_checkpoint import save_checkpoint

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=cfg.data_dir,
        subjects_train=cfg.subjects_train,
        subjects_test=cfg.subjects_test,
        include_extra=cfg.include_extra,
        img_size=cfg.img_size_small,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        mobilenet_norm=False,
    )

    model, loss_fn, optimizer = get_fresh_model("cnn_scratch", cfg, device,
                                                num_classes=len(classes))

    best_val = 0.0
    patience = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_ep(model, train_loader, optimizer, loss_fn, device)
        va = infer_one_ep(model, val_loader, loss_fn, device, validation=True)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                                          **{f"val_{k}": v for k, v in va.items()}})
        print(f"[ep {epoch:02d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f}  "
              f"| val loss={va['loss']:.4f} acc={va['acc']:.4f}")

        if va["acc"] > best_val:
            best_val = va["acc"]
            patience = 0
            save_checkpoint(run_dir / "models" / "best.pt",
                            model, optimizer, epoch=epoch, val_acc=va["acc"])
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[early-stop] no improvement for {patience} epochs.")
                break

    test = infer_one_ep(model, test_loader, loss_fn, device, validation=False)
    return {"model": "cnn_scratch",
            "num_classes": len(classes),
            "classes": classes,
            "best_val_acc": best_val,
            "test": test,
            "history": history}


def run_mobilenetv2(cfg, run_dir: Path, device: torch.device) -> dict:
    raise NotImplementedError(
        "MobileNetV2 con progressive unfreezing lo implementa Yeray (semana 2).\n"
        "Esqueleto: tres fases (head / partial / full) según cfg.mobilenet_phases."
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    if cfg.model is None:
        print("ERROR: --model is required.", file=sys.stderr)
        return 2

    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    run_dir = _make_run_dir(cfg.runs_dir, cfg.model)
    _save_params(run_dir, cfg)
    print(f"[main] device={device}  run_dir={run_dir}")

    if cfg.model == "dummy":
        scores = run_dummy(cfg, run_dir)
    elif cfg.model == "gabor_svm":
        scores = run_gabor_svm(cfg, run_dir)
    elif cfg.model == "cnn_scratch":
        scores = run_cnn_scratch(cfg, run_dir, device)
    elif cfg.model == "mobilenetv2":
        scores = run_mobilenetv2(cfg, run_dir, device)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    _save_scores(run_dir, scores)
    print(f"[main] done. scores written to {run_dir / 'scores' / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
