"""
T6 — Analisis de errores del modelo CNN scratch (Toni).
Lee el results.json mas reciente de runs/cnn_scratch_*/scores/
y genera:
  - presentation/error_analysis.md
  - figures/confusion_cnn_scratch.png
  - figures/top_confusions_cnn_scratch.png
  - figures/history_cnn_scratch.png

Uso:  python error_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from metrics.confusion_matrix import confusion_matrix
from metrics.per_class import per_class_metrics
from metrics.accuracy import accuracy


def latest_run(model="cnn_scratch"):
    runs = sorted(Path("runs").glob(f"{model}_*"))
    if not runs:
        raise FileNotFoundError(
            f"No se encontro runs/{model}_*. "
            f"Ejecuta primero: python main.py --model {model} --epochs 25"
        )
    return runs[-1]


def load_results(run_dir):
    with open(run_dir / "scores" / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("CNN Scratch - Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix: {out_path}")


def plot_top_confusions(pairs, out_path, top_n=10):
    pairs = pairs[:top_n]
    labels = [f"{p['true']}->{p['pred']}" for p in pairs]
    counts = [p["count"] for p in pairs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(labels)), counts, color=sns.color_palette("Reds_r", len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Num confusiones", fontsize=12)
    ax.set_title("CNN Scratch - Top confusiones (true -> predicted)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top confusions: {out_path}")


def plot_history(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "s-", label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  History: {out_path}")


def generate_md(results, cm, classes, top_pairs, pc, out_path):
    test = results["test"]
    f1_scores = pc["f1"]
    class_f1 = sorted(zip(classes, f1_scores), key=lambda x: x[1])

    md = [
        "# Analisis de errores - CNN Scratch", "",
        f"**Test accuracy global:** {test['acc']:.4f} ({test['acc']*100:.1f}%)",
        f"**Numero de clases:** {len(classes)}",
        f"**Total muestras test:** {len(test['y_true'])}", "",
        "## Top confusiones mas frecuentes", "",
        "| # | True | Predicted | Errores | Notas |",
        "|---|------|-----------|---------|-------|",
    ]
    known = {
        ("M","N"): "Formas muy similares (dedos sobre pulgar)",
        ("N","M"): "Formas muy similares (dedos sobre pulgar)",
        ("S","A"): "Puno cerrado, diferencia sutil del pulgar",
        ("A","S"): "Puno cerrado, diferencia sutil del pulgar",
        ("U","V"): "Dos dedos, varia la separacion",
        ("V","U"): "Dos dedos, varia la separacion",
        ("R","U"): "Dedos cruzados vs paralelos",
        ("U","R"): "Dedos cruzados vs paralelos",
    }
    for i, p in enumerate(top_pairs[:10], 1):
        note = known.get((p["true"], p["pred"]), "")
        md.append(f"| {i} | **{p['true']}** | **{p['pred']}** | {p['count']} | {note} |")

    md += ["", "## Clases con peor F1-score", "",
           "| Clase | F1 | Precision | Recall | Support |",
           "|-------|----|-----------|--------|---------|"]
    for cls, f1 in class_f1[:5]:
        idx = classes.index(cls)
        md.append(f"| **{cls}** | {f1:.3f} | {pc['precision'][idx]:.3f} | "
                  f"{pc['recall'][idx]:.3f} | {pc['support'][idx]} |")

    md += ["", "## Interpretacion", "",
           "Las confusiones tipicas en ASL fingerspelling:",
           "1. **M/N**: diferencia en num dedos sobre pulgar, sutil a 64x64.",
           "2. **S/A/E**: variantes de puno cerrado.",
           "3. **U/V/R**: dos dedos extendidos con variaciones.",
           "", "### Posibles mejoras", "",
           "- Mayor resolucion (224x224) con transfer learning.",
           "- Data augmentation mas agresiva.",
           "- Attention mechanism para focalizar en dedos.", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Error analysis MD: {out_path}")


def main():
    print("=== T6 - Analisis de errores CNN Scratch ===\n")
    run_dir = latest_run("cnn_scratch")
    print(f"Run: {run_dir}")
    results = load_results(run_dir)
    classes = results["classes"]
    y_true, y_pred = results["test"]["y_true"], results["test"]["y_pred"]
    num_classes = results["num_classes"]
    print(f"Test acc: {results['test']['acc']:.4f}, Clases: {num_classes}\n")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    pc = per_class_metrics(y_true, y_pred, num_classes)

    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": classes[i], "pred": classes[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda x: x["count"], reverse=True)

    print("Top 5 confusiones:")
    for p in pairs[:5]:
        print(f"  {p['true']} -> {p['pred']}: {p['count']}")

    Path("figures").mkdir(exist_ok=True)
    plot_confusion(cm, classes, "figures/confusion_cnn_scratch.png")
    plot_top_confusions(pairs, "figures/top_confusions_cnn_scratch.png")
    if "history" in results and results["history"]:
        plot_history(results["history"], "figures/history_cnn_scratch.png")
    generate_md(results, cm, classes, pairs, pc, Path("presentation/error_analysis.md"))
    print("\n=== T6 COMPLETADA ===")


if __name__ == "__main__":
    main()
