import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # modo sin pantalla: necesario en Colab y servidores sin display
import matplotlib.pyplot as plt
import numpy as np

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
    colors = {"head": "#2196F3", "partial": "#FF9800", "full": "#4CAF50"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MobileNetV2 — Training History by Phase", fontsize=13, fontweight="bold")

    # Agrupamos datos por fase manteniendo el orden de aparición
    phases_order = []
    phases = {}
    global_epoch = 0
    for h in history:
        global_epoch += 1
        phase = h["phase"]
        if phase not in phases:
            phases[phase] = {"x": [], "train_loss": [], "val_loss": [],
                             "train_acc": [], "val_acc": []}
            phases_order.append(phase)
        phases[phase]["x"].append(global_epoch)
        phases[phase]["train_loss"].append(h["train_loss"])
        phases[phase]["val_loss"].append(h["val_loss"])
        phases[phase]["train_acc"].append(h["train_acc"])
        phases[phase]["val_acc"].append(h["val_acc"])

    for i, phase in enumerate(phases_order):
        d = phases[phase]
        color = colors.get(phase, "gray")

        # Añadimos el último punto de la fase anterior al inicio de esta
        # para que las líneas se toquen y el color cambie justo en la unión
        if i > 0:
            prev = phases[phases_order[i - 1]]
            x          = [prev["x"][-1]]          + d["x"]
            train_loss = [prev["train_loss"][-1]]  + d["train_loss"]
            val_loss   = [prev["val_loss"][-1]]    + d["val_loss"]
            train_acc  = [prev["train_acc"][-1]]   + d["train_acc"]
            val_acc    = [prev["val_acc"][-1]]     + d["val_acc"]
        else:
            x, train_loss, val_loss = d["x"], d["train_loss"], d["val_loss"]
            train_acc, val_acc = d["train_acc"], d["val_acc"]

        axes[0].plot(x, train_loss, "o-",  color=color, markersize=4, label=f"train ({phase})")
        axes[0].plot(x, val_loss,   "s--", color=color, markersize=4, alpha=0.7, label=f"val ({phase})")
        axes[1].plot(x, train_acc,  "o-",  color=color, markersize=4, label=f"train ({phase})")
        axes[1].plot(x, val_acc,    "s--", color=color, markersize=4, alpha=0.7, label=f"val ({phase})")

    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Global Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  History MobileNetV2: {out_path}")


def plot_gabor_per_class(results, out_path):
    """Accuracy por clase del Gabor+SVM — sustituye las curvas de entrenamiento."""
    from metrics.per_class import per_class_metrics
    test = results["test"]
    classes = results["classes"]
    pc = per_class_metrics(test["y_true"], test["y_pred"], results["num_classes"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    ax.bar(x, pc["f1"], color=[
        "#4CAF50" if v >= 0.5 else "#FF7043" for v in pc["f1"]
    ], edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Gabor + SVM — F1-score por clase (test subject-5)",
                 fontsize=13, fontweight="bold")
    ax.axhline(np.mean(pc["f1"]), color="gray", linestyle="--",
               linewidth=1, label=f"F1 macro medio: {np.mean(pc['f1']):.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gabor per-class F1: {out_path}")


def plot_comparison_bar(rows, out_path):
    """Gráfico de barras agrupadas: un grupo por métrica, una barra por modelo."""
    metrics = ["acc", "precision_macro", "recall_macro", "f1_macro"]
    labels  = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
    model_colors  = {"gabor_svm": "#FF7043", "cnn_scratch": "#42A5F5", "mobilenetv2": "#66BB6A"}
    model_display = {"gabor_svm": "Gabor + SVM", "cnn_scratch": "Custom CNN",
                     "mobilenetv2": "MobileNetV2"}

    n_metrics = len(metrics)
    n_models  = len(rows)
    bar_width = 0.22
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(rows):
        offset = (i - n_models / 2 + 0.5) * bar_width
        values = [row.get(m, 0) or 0 for m in metrics]
        bars = ax.bar(x + offset, values, bar_width,
                      label=model_display.get(row["model"], row["model"]),
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
    ax.set_title("Model Comparison — ASL Fingerspelling", fontsize=13, fontweight="bold")
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

        if model == "mobilenetv2" and results.get("history"):
            plot_mobilenet_history(results["history"], "figures/history_mobilenetv2.png")

        if model == "gabor_svm":
            plot_gabor_per_class(results, "figures/gabor_per_class_f1.png")

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
