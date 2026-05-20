"""Argparse + JSON config. Precedence: CLI > JSON > default.json."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CONFIG = Path(__file__).parent / "doc" / "default.json"
MODEL_CHOICES = ("gabor_svm", "cnn_scratch", "mobilenetv2")


def to_namespace(d):
    out = SimpleNamespace()
    for k, v in d.items():
        setattr(out, k, to_namespace(v) if isinstance(v, dict) else v)
    return out


def build_parser():
    p = argparse.ArgumentParser(description="ASL Fingerspelling")
    p.add_argument("--model", choices=MODEL_CHOICES, required=False)
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--runs_dir", type=str, default=None)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    for key in ("epochs", "batch_size", "lr", "seed", "data_dir", "runs_dir", "num_workers"):
        val = getattr(args, key)
        if val is not None:
            cfg_dict[key] = val

    cfg_dict["model"] = args.model
    cfg_dict["device"] = args.device
    cfg_dict["debug"] = args.debug
    cfg_dict["config_path"] = str(config_path)

    return to_namespace(cfg_dict)
