import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # modo sin pantalla: necesario en Colab y servidores sin display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from metrics.confusion_matrix import confusion_matrix
from metrics.per_class import per_class_metrics
from visual.plot import plot_confusion_matrix

MODELS = ["gabor_svm", "cnn_scratch", "mobilenetv2"]

def latest_run(model):
    """Devuelve la carpeta runs/{model}_* más reciente, o None si no existe."""
    runs = sorted(Path("runs").glob(f"{model}_*"))
    return runs[-1] if runs else None


def load_results(run_dir):
    with open(run_dir / "scores" / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_mobilenet_history(history, out_path):
    """Curvas de loss y accuracy del MobileNetV2 con las 3 fases coloreadas."""
    # Colores distintos para cada fase para que se vea visualmente el progressive unfreezing
    colors = {"head": "#2196F3", "partial": "#FF9800", "full": "#4CAF50"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MobileNetV2 — training history por fase", fontsize=13, fontweight="bold")

    # Reconstruimos el eje x como número de época global (no reinicia por fase)
    global_epoch = 0
    prev_phase = None
    x_ticks, x_labels = [], []

    for h in history:
        if h["phase"] != prev_phase:
            global_epoch = global_epoch  # la época global sigue avanzando
            prev_phase = h["phase"]
        global_epoch += 1

        color = colors.get(h["phase"], "gray")
        label_tr = f"train ({h['phase']})" if h["epoch"] == 1 else None
        label_va = f"val ({h['phase']})"   if h["epoch"] == 1 else None

        axes[0].plot(global_epoch, h["train_loss"], "o", color=color, markersize=4, label=label_tr)
        axes[0].plot(global_epoch, h["val_loss"],   "s", color=color, markersize=4,
                     alpha=0.6, label=label_va)
        axes[1].plot(global_epoch, h["train_acc"],  "o", color=color, markersize=4, label=label_tr)
        axes[1].plot(global_epoch, h["val_acc"],    "s", color=color, markersize=4,
                     alpha=0.6, label=label_va)

    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Época global")
        ax.grid(alpha=0.3)
        # Evitamos duplicados en la leyenda
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  History MobileNetV2: {out_path}")


def plot_comparison_bar(rows, out_path):
    """Gráfico de barras agrupadas: un grupo por métrica, una barra por modelo."""
    metrics = ["acc", "precision_macro", "recall_macro", "f1_macro"]
    labels  = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
    model_colors = {"gabor_svm": "#FF7043", "cnn_scratch": "#42A5F5", "mobilenetv2": "#66BB6A"}

    n_metrics = len(metrics)
    n_models  = len(rows)
    bar_width = 0.22
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(rows):
        # Offset para que las barras de cada modelo no se solapen
        offset = (i - n_models / 2 + 0.5) * bar_width
        values = [row.get(m, 0) or 0 for m in metrics]
        bars = ax.bar(x + offset, values, bar_width,
                      label=row["model"],
                      color=model_colors.get(row["model"], "gray"),
                      edgecolor="white", linewidth=0.8)
        # Valor encima de cada barra
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de modelos — ASL Fingerspelling", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison bar chart: {out_path}")

def main():
    print("=== Y4 — Comparación final de modelos ===\n")
    Path("figures").mkdir(exist_ok=True)

    rows = []

    for model in MODELS:
        run = latest_run(model)
        if run is None:
            print(f"[SKIP] {model}: no hay runs/ todavia.")
            continue

        results = load_results(run)
        classes   = results["classes"]
        test      = results["test"]
        n_classes = results["num_classes"]

        # Matriz de confusión y figura
        cm = confusion_matrix(test["y_true"], test["y_pred"], n_classes)
        plot_confusion_matrix(cm, classes, f"figures/confusion_{model}.png")

        # Métricas macro (gabor_svm ya las tiene; para CNN/MobileNet las calculamos)
        pc = per_class_metrics(test["y_true"], test["y_pred"], n_classes)
        prec = test.get("precision_macro") or float(np.mean(pc["precision"]))
        rec  = test.get("recall_macro")    or float(np.mean(pc["recall"]))
        f1   = test.get("f1_macro")        or float(np.mean(pc["f1"]))

        rows.append({
            "model":            model,
            "acc":              test["acc"],
            "precision_macro":  prec,
            "recall_macro":     rec,
            "f1_macro":         f1,
        })

        # Curvas de entrenamiento solo para MobileNetV2 (tiene historial por fases)
        if model == "mobilenetv2" and results.get("history"):
            plot_mobilenet_history(results["history"], "figures/history_mobilenetv2.png")

    if not rows:
        print("No hay resultados todavia. Entrena al menos un modelo primero.")
        return

    # Tabla por consola
    print("\n" + "=" * 58)
    print(f"{'Model':<16}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 58)
    for r in rows:
        print(f"{r['model']:<16}{r['acc']:>8.3f}"
              f"{r['precision_macro']:>8.3f}"
              f"{r['recall_macro']:>8.3f}"
              f"{r['f1_macro']:>8.3f}")
    print("=" * 58)

    # Gráfico de barras comparativo
    plot_comparison_bar(rows, "figures/comparison_bar.png")
    print("\n=== Y4 COMPLETADA ===")
    print("Figuras guardadas en figures/")


if __name__ == "__main__":
    main()
