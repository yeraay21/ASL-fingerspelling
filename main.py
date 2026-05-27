"""Entry point. Usage: python main.py --model {gabor_svm|cnn_scratch|mobilenetv2}"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from params import parse_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_run_dir(runs_dir, model):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = Path(runs_dir) / f"{model}_{stamp}"
    for sub in ("models", "outputs", "scores", "logs", "parameters"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    return run


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda o: vars(o))



def run_gabor_svm(cfg, run_dir: Path) -> dict:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from tqdm import tqdm

    from dataset.dataset import ASLDataset, discover_classes
    from dataset.preprocessing.image_processing import (
        read_image_bgr, to_grayscale, resize, to_float01,
    )

    # load functions implemented in filters.py
    from dataset.preprocessing.filters import build_gabor_bank, extract_features

    classes = discover_classes(cfg.data_dir, cfg.subjects_train[0])

    def load_gray(subjects, include_extra):
        ds = ASLDataset(cfg.data_dir, subjects, classes,
                        transform=lambda img: img,
                        include_extra=include_extra)
        X = np.zeros((len(ds), cfg.img_size_small, cfg.img_size_small), dtype=np.float32)
        y = np.zeros(len(ds), dtype=np.int64)

        # go through the dataset to prepare our data
        for i, (img_rgb, label) in enumerate(tqdm(ds, desc="loading grayscale images")):
            # grayscaling our images + resizing
            gray = to_grayscale(img_rgb)
            gray = resize(gray, cfg.img_size_small)

            # fill the table X and y
            X[i] = to_float01(gray)
            y[i] = label
        return X, y

    # apply load_gray on test set and training set
    X_tr, y_tr = load_gray(cfg.subjects_train, cfg.include_extra)
    X_te, y_te = load_gray(cfg.subjects_test, include_extra=False)

    # use the functions from filters.py to get a vector with 80 columns (mean+std for 40 gabor filters)
    bank = build_gabor_bank(cfg.gabor.frequencies, cfg.gabor.orientations)
    F_tr = extract_features(X_tr, bank)
    F_te = extract_features(X_te, bank)

    # standardisation of the dataset 
    scaler = StandardScaler().fit(F_tr)
    F_tr_s, F_te_s = scaler.transform(F_tr), scaler.transform(F_te)

    # apply PCA to reduce colinearity, cleaner data
    pca = PCA(n_components=cfg.gabor.pca_components, random_state=cfg.seed).fit(F_tr_s)
    Z_tr, Z_te = pca.transform(F_tr_s), pca.transform(F_te_s)

    # training model, GridSearch useful to automatically find the best C value
    grid = GridSearchCV(
        SVC(kernel=cfg.gabor.svm_kernel),
        param_grid={"C": cfg.gabor.svm_C_grid},
        cv=3, n_jobs=2,
    ).fit(Z_tr, y_tr)

    # evaluation of the performance
    y_pred = grid.best_estimator_.predict(Z_te)
    acc = float(accuracy_score(y_te, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)

    return {"model": "gabor_svm",
            "num_classes": len(classes), "classes": classes,
            "best_C": grid.best_params_["C"], "cv_best_score": float(grid.best_score_),
            "test": {"acc": acc, "precision_macro": float(p),
                     "recall_macro": float(r), "f1_macro": float(f),
                     "y_true": y_te.tolist(), "y_pred": y_pred.tolist()}}


def run_cnn_scratch(cfg, run_dir, device):
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

    model, loss_fn, optimizer = get_fresh_model("cnn_scratch", cfg, device, num_classes=len(classes))

    best_val = 0.0
    patience = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_ep(model, train_loader, optimizer, loss_fn, device)
        va = infer_one_ep(model, val_loader, loss_fn, device, validation=True)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"ep {epoch:02d} | train loss={tr['loss']:.4f} acc={tr['acc']:.4f} "
              f"| val loss={va['loss']:.4f} acc={va['acc']:.4f}")

        if va["acc"] > best_val:
            best_val = va["acc"]
            patience = 0
            save_checkpoint(run_dir / "models" / "best.pt", model, optimizer,
                            epoch=epoch, val_acc=va["acc"])
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"early stop: {patience} epochs sin mejorar")
                break

    test = infer_one_ep(model, test_loader, loss_fn, device, validation=False)
    return {"model": "cnn_scratch", "num_classes": len(classes), "classes": classes,
            "best_val_acc": best_val, "test": test, "history": history}


def run_mobilenetv2(cfg, run_dir, device):
    import torch
    from dataset.loaders import get_loaders
    from utility.get_fresh_model import get_fresh_model
    from utility.save_checkpoint import save_checkpoint
    from phases.train import train_one_ep
    from phases.infer import infer_one_ep

    # Loaders con imágenes 224x224 RGB y normalización ImageNet (requerimiento de MobileNetV2)
    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=cfg.data_dir,
        subjects_train=cfg.subjects_train,
        subjects_test=cfg.subjects_test,
        include_extra=cfg.include_extra,
        img_size=cfg.img_size_mobile,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        mobilenet_norm=True,  # activa mean/std de ImageNet en lugar de [0,1]
    )

    # Creamos el modelo con la cabeza adaptada a nuestras 24 clases
    model, loss_fn, _ = get_fresh_model("mobilenetv2", cfg, device, num_classes=len(classes))

    history = []

    # Iteramos sobre las 3 fases definidas en default.json: head → partial → full
    for phase in cfg.mobilenet_phases:

        # Congela o descongela capas del backbone según la fase
        model.set_phase(phase.unfreeze_from)

        # Solo pasamos al optimizador los parámetros que tienen requires_grad=True
        # (los congelados no se actualizan y así ahorramos tiempo y memoria)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=phase.lr,
        )

        for epoch in range(1, phase.epochs + 1):
            tr = train_one_ep(model, train_loader, optimizer, loss_fn, device)
            va = infer_one_ep(model, val_loader, loss_fn, device, validation=True)

            # Guardamos la métrica de cada época en el historial para las curvas de plot
            history.append({
                "phase": phase.name,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items() if k not in ("y_true", "y_pred")},
                **{f"val_{k}":   v for k, v in va.items() if k not in ("y_true", "y_pred")},
            })
            print(f"[{phase.name} ep {epoch:02d}] train acc={tr['acc']:.3f} | val acc={va['acc']:.3f}")

        # Guardamos el checkpoint al final de cada fase (por si Colab se cae)
        save_checkpoint(
            run_dir / "models" / f"{phase.name}.pt",
            model, optimizer,
            phase=phase.name, val_acc=va["acc"],
        )

    # Evaluación final en el test set (subject-5) — solo se hace una vez al terminar
    test = infer_one_ep(model, test_loader, loss_fn, device, validation=False)

    return {
        "model": "mobilenetv2",
        "num_classes": len(classes),
        "classes": classes,
        "test": test,
        "history": history,
    }


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.model is None:
        print("ERROR: --model is required.", file=sys.stderr)
        return 2

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = make_run_dir(cfg.runs_dir, cfg.model)
    save_json(run_dir / "parameters" / "config.json", cfg)
    print(f"device={device}  run_dir={run_dir}")

    if cfg.model == "gabor_svm":
        scores = run_gabor_svm(cfg, run_dir)
    elif cfg.model == "cnn_scratch":
        scores = run_cnn_scratch(cfg, run_dir, device)
    elif cfg.model == "mobilenetv2":
        scores = run_mobilenetv2(cfg, run_dir, device)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    save_json(run_dir / "scores" / "results.json", scores)
    print(f"done. scores -> {run_dir / 'scores' / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
