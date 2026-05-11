"""
Regresion Logistica para clasificacion de nivel_riesgo.
Obligatoria segun enunciado. Requiere StandardScaler.
Compara regularizacion L1 (Lasso) y L2 (Ridge) con C optimo en val.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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


def plot_coeficientes(model, feature_cols, ruta):
    """Muestra coeficientes de cada clase para identificar features clave."""
    coefs = model.coef_
    n_clases = len(model.classes_)
    fig, axes = plt.subplots(1, n_clases, figsize=(5 * n_clases, 6))
    if n_clases == 1:
        axes = [axes]
    for i, (ax, clase) in enumerate(zip(axes, [CLASES[c] for c in model.classes_])):
        idx = np.argsort(np.abs(coefs[i]))[-12:]
        ax.barh(range(len(idx)), coefs[i][idx], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_cols[j] for j in idx], fontsize=8)
        ax.axvline(0, color="gray", linestyle="--", lw=0.8)
        ax.set_title(f"Clase {clase}", fontsize=10)
        ax.set_xlabel("Coeficiente")
    plt.suptitle("Coeficientes Regresión Logística (top 12 por clase)", fontsize=11)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
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
    print("  LOGISTIC REGRESSION — Clasificacion nivel_riesgo")
    print("=" * 55)

    data = preparar(verbose=False)
    X_train_sc  = data["X_train_sc"]
    X_val_sc    = data["X_val_sc"]
    X_test_sc   = data["X_test_sc"]
    y_train     = data["y_train"]
    y_val       = data["y_val"]
    y_test      = data["y_test"]
    feature_cols = data["feature_cols"]

    # ── Búsqueda de C y regularización ───────────────────────────────────────
    Cs = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    penalties = ["l2", "l1"]
    mejores = []

    print(f"\n{'Penalty':<10} {'C':>8} {'F1-Macro(val)':>14} {'F1-ALTO(val)':>13}")
    for pen in penalties:
        solver = "saga"
        for C in Cs:
            lr = LogisticRegression(
                penalty=pen, C=C, solver=solver,
                class_weight="balanced",
                max_iter=2000, random_state=42,
            )
            lr.fit(X_train_sc, y_train)
            y_pred = lr.predict(X_val_sc)
            f1m = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1a = f1_score(y_val, y_pred, labels=[2], average=None, zero_division=0)[0]
            mejores.append((f1m, pen, C, lr))
            print(f"{pen:<10} {C:>8.2f} {f1m:>14.3f} {f1a:>13.3f}")

    mejores.sort(key=lambda x: -x[0])
    f1_best, pen_best, C_best, lr_best = mejores[0]
    print(f"\nMejor: penalty={pen_best}, C={C_best}, F1-Macro(val)={f1_best:.3f}")

    # ── Evaluación final ──────────────────────────────────────────────────────
    y_pred_val  = lr_best.predict(X_val_sc)
    y_pred_test = lr_best.predict(X_test_sc)

    metricas_val  = evaluar(f"LR ({pen_best}, C={C_best})", y_val,  y_pred_val,  "VAL")
    metricas_test = evaluar(f"LR ({pen_best}, C={C_best})", y_test, y_pred_test, "TEST")

    # ── Figuras ───────────────────────────────────────────────────────────────
    plot_confusion(y_val,  y_pred_val,
                   f"Logistic Regression — Validación",
                   FIGS / "lr_confusion_val.png")
    plot_confusion(y_test, y_pred_test,
                   f"Logistic Regression — Test",
                   FIGS / "lr_confusion_test.png")
    plot_coeficientes(lr_best, feature_cols, FIGS / "lr_coeficientes.png")

    print("\n── Resumen ────────────────────────────────────────────")
    print(f"{'Métrica':<15} {'Val':>8} {'Test':>8}")
    for k_m in ["f1_macro", "f1_alto", "kappa"]:
        print(f"{k_m:<15} {metricas_val[k_m]:>8.3f} {metricas_test[k_m]:>8.3f}")

    joblib.dump(lr_best, MDIR / "logistic_regression.pkl")
    print(f"\nModelo guardado: models/logistic_regression.pkl")

    return lr_best, data


if __name__ == "__main__":
    main()
