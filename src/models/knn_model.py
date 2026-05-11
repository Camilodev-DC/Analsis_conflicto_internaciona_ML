"""
K-Nearest Neighbors para clasificacion de nivel_riesgo.
Obligatorio segun enunciado. Requiere StandardScaler.
Busca k optimo en [3, 5, 7, 9, 11] con F1-Macro en val.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, cohen_kappa_score, ConfusionMatrixDisplay,
)
from pathlib import Path
import joblib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.preparar_dataset import preparar

MDIR = Path(__file__).parent.parent.parent / "models"
FIGS = Path(__file__).parent.parent.parent / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)

CLASES = ["BAJO", "MEDIO", "ALTO"]


def evaluar(nombre, y_true, y_pred, split):
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    f1_alto  = f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    clases_p = sorted(np.unique(np.concatenate([y_true, y_pred])))
    nombres_p = [CLASES[i] for i in clases_p]
    print(f"\n{'─'*50}")
    print(f"  {nombre} — {split}")
    print(f"  F1-Macro : {f1_macro:.3f}")
    print(f"  F1-ALTO  : {f1_alto:.3f}")
    print(f"  Kappa    : {kappa:.3f}")
    print(classification_report(y_true, y_pred,
                                labels=clases_p, target_names=nombres_p,
                                zero_division=0))
    return {"f1_macro": f1_macro, "f1_alto": f1_alto, "kappa": kappa}


def plot_k_search(ks, f1s, k_opt, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, f1s, "o-", color="steelblue", lw=2)
    ax.axvline(k_opt, color="crimson", linestyle="--", label=f"k={k_opt} óptimo")
    ax.set_xlabel("k (vecinos)")
    ax.set_ylabel("F1-Macro (val)")
    ax.set_title("Búsqueda de k — KNN")
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def plot_confusion(y_true, y_pred, titulo, ruta):
    clases_p = sorted(np.unique(np.concatenate([y_true, y_pred])))
    nombres_p = [CLASES[i] for i in clases_p]
    cm = confusion_matrix(y_true, y_pred, labels=clases_p)
    disp = ConfusionMatrixDisplay(cm, display_labels=nombres_p)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titulo, fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def main():
    print("=" * 55)
    print("  KNN — Clasificacion nivel_riesgo Hormuz 2024")
    print("=" * 55)

    data = preparar(verbose=False)
    X_train_sc = data["X_train_sc"]
    X_val_sc   = data["X_val_sc"]
    X_test_sc  = data["X_test_sc"]
    y_train    = data["y_train"]
    y_val      = data["y_val"]
    y_test     = data["y_test"]

    # ── Búsqueda de k ────────────────────────────────────────────────────────
    ks = [3, 5, 7, 9, 11, 15]
    f1s = []
    print(f"\n{'k':>4} {'F1-Macro(val)':>14} {'F1-ALTO(val)':>13}")
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        knn.fit(X_train_sc, y_train)
        y_pred = knn.predict(X_val_sc)
        f1m = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1a = f1_score(y_val, y_pred, labels=[2], average=None, zero_division=0)[0]
        f1s.append(f1m)
        print(f"{k:>4} {f1m:>14.3f} {f1a:>13.3f}")

    k_opt = ks[np.argmax(f1s)]
    print(f"\nk óptimo: {k_opt}")

    plot_k_search(ks, f1s, k_opt, FIGS / "knn_k_search.png")

    # ── Modelo final ──────────────────────────────────────────────────────────
    knn_best = KNeighborsClassifier(n_neighbors=k_opt, weights="distance", metric="euclidean")
    knn_best.fit(X_train_sc, y_train)

    y_pred_val  = knn_best.predict(X_val_sc)
    y_pred_test = knn_best.predict(X_test_sc)

    metricas_val  = evaluar(f"KNN (k={k_opt})", y_val,  y_pred_val,  "VAL")
    metricas_test = evaluar(f"KNN (k={k_opt})", y_test, y_pred_test, "TEST")

    plot_confusion(y_val,  y_pred_val,
                   f"KNN k={k_opt} — Validación",
                   FIGS / "knn_confusion_val.png")
    plot_confusion(y_test, y_pred_test,
                   f"KNN k={k_opt} — Test",
                   FIGS / "knn_confusion_test.png")

    print("\n── Resumen ────────────────────────────────────────────")
    print(f"{'Métrica':<15} {'Val':>8} {'Test':>8}")
    for k_m in ["f1_macro", "f1_alto", "kappa"]:
        print(f"{k_m:<15} {metricas_val[k_m]:>8.3f} {metricas_test[k_m]:>8.3f}")

    joblib.dump(knn_best, MDIR / "knn.pkl")
    print(f"\nModelo guardado: models/knn.pkl")

    return knn_best, data


if __name__ == "__main__":
    main()
