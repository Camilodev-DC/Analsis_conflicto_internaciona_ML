"""
Tabla comparativa de todos los modelos sobre el set de validacion.
Genera figura resumen y tabla markdown para el informe.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.preparar_dataset import preparar

import joblib

MDIR = Path(__file__).parent.parent.parent / "models"
FIGS = Path(__file__).parent.parent.parent / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)


def metricas(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_alto":  f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0],
        "kappa":    cohen_kappa_score(y_true, y_pred),
        "accuracy": (y_true == y_pred).mean(),
    }


def main():
    data = preparar(verbose=False)
    X_val_sc = data["X_val_sc"]
    y_val    = data["y_val"]

    modelos = {
        "Naive Bayes":          MDIR / "naive_bayes.pkl",
        "KNN (k=5)":            MDIR / "knn.pkl",
        "Logistic Reg (L1)":    MDIR / "logistic_regression.pkl",
    }

    resultados = []
    for nombre, ruta in modelos.items():
        if not ruta.exists():
            print(f"  [SKIP] {nombre} — modelo no encontrado")
            continue
        modelo = joblib.load(ruta)
        y_pred = modelo.predict(X_val_sc)
        m = metricas(y_val, y_pred)
        resultados.append({"Modelo": nombre, **m})
        print(f"{nombre:<25} F1-Macro={m['f1_macro']:.3f}  F1-ALTO={m['f1_alto']:.3f}  Kappa={m['kappa']:.3f}")

    # ── Figura comparativa ────────────────────────────────────────────────────
    metricas_cols = ["f1_macro", "f1_alto", "kappa"]
    etiquetas     = ["F1-Macro", "F1-ALTO", "Kappa"]
    nombres       = [r["Modelo"] for r in resultados]
    x = np.arange(len(metricas_cols))
    w = 0.25
    colores = ["steelblue", "darkorange", "crimson", "seagreen"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (res, color) in enumerate(zip(resultados, colores)):
        vals = [res[m] for m in metricas_cols]
        offset = (i - len(resultados) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=res["Modelo"], color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparación de modelos — Set de Validación", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGS / "comparacion_modelos.png", dpi=150)
    plt.close(fig)
    print(f"\n  Figura guardada: comparacion_modelos.png")

    # ── Tabla markdown ────────────────────────────────────────────────────────
    print("\n## Tabla comparativa (validación)\n")
    print(f"| {'Modelo':<25} | {'F1-Macro':>9} | {'F1-ALTO':>8} | {'Kappa':>7} | {'Accuracy':>9} |")
    print(f"|{'-'*27}|{'-'*11}|{'-'*10}|{'-'*9}|{'-'*11}|")
    for r in sorted(resultados, key=lambda x: -x["f1_macro"]):
        print(f"| {r['Modelo']:<25} | {r['f1_macro']:>9.3f} | {r['f1_alto']:>8.3f} | {r['kappa']:>7.3f} | {r['accuracy']:>9.3f} |")


if __name__ == "__main__":
    main()
