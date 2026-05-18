"""Argument parsing + config loading for main.py.

Precedence: CLI flag > JSON config > default.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CONFIG = Path(__file__).parent / "doc" / "default.json"

MODEL_CHOICES = ("gabor_svm", "cnn_scratch", "mobilenetv2", "dummy")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursive dict -> SimpleNamespace so we can use cfg.field.subfield."""
    out = SimpleNamespace()
    for k, v in d.items():
        setattr(out, k, _to_namespace(v) if isinstance(v, dict) else v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASL Fingerspelling — unified entry point.")
    p.add_argument("--model", choices=MODEL_CHOICES, required=False,
                   help="Which model to train.")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help="Path to JSON config file.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--runs_dir", type=str, default=None)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Run with a tiny subset to check the pipeline end-to-end.")
    return p


def parse_args(argv: list[str] | None = None) -> SimpleNamespace:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg_dict = _load_json(config_path)

    for cli_key in ("epochs", "batch_size", "lr", "seed",
                    "data_dir", "runs_dir", "num_workers"):
        cli_val = getattr(args, cli_key)
        if cli_val is not None:
            cfg_dict[cli_key] = cli_val

    cfg_dict["model"] = args.model
    cfg_dict["device"] = args.device
    cfg_dict["debug"] = args.debug
    cfg_dict["config_path"] = str(config_path)

    return _to_namespace(cfg_dict)
