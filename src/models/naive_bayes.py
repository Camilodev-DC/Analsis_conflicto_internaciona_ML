"""
Naive Bayes para clasificacion de nivel_riesgo (BAJO / MEDIO / ALTO).
Dos variantes comparadas:
  1. GaussianNB con prior ajustado (baseline)
  2. ComplementNB — robusto a la violacion de independencia por features correlacionadas
     El modelo final es ComplementNB por mayor F1-Macro (0.509 vs 0.440).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, cohen_kappa_score, ConfusionMatrixDisplay,
)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.preparar_dataset import preparar

MDIR = Path(__file__).parent.parent.parent / "models"
FIGS = Path(__file__).parent.parent.parent / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)

CLASES = ["BAJO", "MEDIO", "ALTO"]
LABEL_MAP_INV = {0: "BAJO", 1: "MEDIO", 2: "ALTO"}


def calcular_priors_balanceados(y_train):
    """Prior inverso a la frecuencia para compensar el desbalance."""
    clases, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts
    priors = weights / weights.sum()
    ordered = np.zeros(3)
    for c, p in zip(clases, priors):
        ordered[c] = p
    return ordered


def evaluar(nombre, y_true, y_pred, split):
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    f1_alto  = f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    clases_presentes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    nombres_presentes = [CLASES[i] for i in clases_presentes]
    print(f"\n{'─'*50}")
    print(f"  {nombre} — {split}")
    print(f"  F1-Macro : {f1_macro:.3f}")
    print(f"  F1-ALTO  : {f1_alto:.3f}")
    print(f"  Kappa    : {kappa:.3f}")
    print(classification_report(y_true, y_pred,
                                labels=clases_presentes,
                                target_names=nombres_presentes,
                                zero_division=0))
    return {"f1_macro": f1_macro, "f1_alto": f1_alto, "kappa": kappa}


def plot_confusion(y_true, y_pred, titulo, ruta):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titulo, fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def plot_probs(model, X_sc, y_true, fechas, split, ruta):
    """Probabilidades predichas por clase a lo largo del tiempo."""
    probs = model.predict_proba(X_sc)
    fig, ax = plt.subplots(figsize=(12, 4))
    colores = ["steelblue", "orange", "crimson"]
    for i, (clase, color) in enumerate(zip(CLASES, colores)):
        ax.plot(fechas, probs[:, i], label=clase, color=color, alpha=0.8, lw=1.5)
    # Marcar días ALTO reales
    alto_mask = y_true == 2
    if alto_mask.any():
        ax.scatter(fechas[alto_mask], np.ones(alto_mask.sum()) * 0.95,
                   marker="v", color="crimson", s=50, zorder=5, label="ALTO real")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Probabilidad")
    ax.set_title(f"Probabilidades Naive Bayes — {split}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def main():
    print("=" * 60)
    print("  NAIVE BAYES — Clasificacion nivel_riesgo Hormuz 2024")
    print("=" * 60)

    data = preparar(verbose=False)
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]
    df_val  = data["df_val"]

    # ── Escalado para cada variante ───────────────────────────────────────────
    # GaussianNB usa StandardScaler (ya en data)
    X_train_sc = data["X_train_sc"]
    X_val_sc   = data["X_val_sc"]
    X_test_sc  = data["X_test_sc"]

    # ComplementNB requiere valores >= 0: MinMaxScaler sobre datos sin escalar
    mms = MinMaxScaler()
    X_train_mm = mms.fit_transform(data["X_train"])
    X_val_mm   = mms.transform(data["X_val"])
    X_test_mm  = mms.transform(data["X_test"])

    # ── Variante 1: GaussianNB con prior ajustado (baseline) ─────────────────
    print("\n── Variante 1: GaussianNB (prior ajustado) ────────────")
    priors = calcular_priors_balanceados(y_train)
    print(f"Priors: BAJO={priors[0]:.3f}  MEDIO={priors[1]:.3f}  ALTO={priors[2]:.3f}")
    gnb = GaussianNB(priors=priors)
    gnb.fit(X_train_sc, y_train)
    met_gnb_val  = evaluar("GaussianNB", y_val,  gnb.predict(X_val_sc),  "VAL")
    met_gnb_test = evaluar("GaussianNB", y_test, gnb.predict(X_test_sc), "TEST")

    # ── Variante 2: ComplementNB (robusto a correlacion entre features) ───────
    # ComplementNB modela el complemento de cada clase en lugar de la clase misma.
    # Esto reduce el efecto de doble conteo cuando features estan correlacionadas
    # (en este dataset, las 5 features GDELT tienen r>0.9 entre si).
    print("\n── Variante 2: ComplementNB (MinMaxScaler) ────────────")
    cnb = ComplementNB()
    cnb.fit(X_train_mm, y_train)
    met_cnb_val  = evaluar("ComplementNB", y_val,  cnb.predict(X_val_mm),  "VAL")
    met_cnb_test = evaluar("ComplementNB", y_test, cnb.predict(X_test_mm), "TEST")

    # ── Seleccion del modelo final ────────────────────────────────────────────
    # ComplementNB: F1-Macro=0.509 vs GaussianNB: F1-Macro=0.440
    # ComplementNB gana en F1-Macro y Kappa; empatan en F1-ALTO
    modelo_final = cnb
    X_val_final  = X_val_mm
    y_pred_final = cnb.predict(X_val_mm)
    nombre_final = "ComplementNB"

    # ── Figuras ───────────────────────────────────────────────────────────────
    plot_confusion(y_val,  gnb.predict(X_val_sc),
                   "GaussianNB — Validación",   FIGS / "nb_gaussian_confusion_val.png")
    plot_confusion(y_val,  cnb.predict(X_val_mm),
                   "ComplementNB — Validación", FIGS / "nb_complement_confusion_val.png")
    plot_confusion(y_test, cnb.predict(X_test_mm),
                   "ComplementNB — Test",       FIGS / "nb_confusion_test.png")
    plot_probs(gnb, X_val_sc, y_val,
               df_val["fecha"].values, "GaussianNB Val", FIGS / "nb_probs_val.png")

    # ── Análisis de errores del modelo final ──────────────────────────────────
    print(f"\n── Errores {nombre_final} en validación ──────────────")
    errores = df_val.copy().reset_index(drop=True)
    errores["pred"] = [LABEL_MAP_INV[p] for p in y_pred_final]
    errores["real"] = [LABEL_MAP_INV[r] for r in y_val]
    errores = errores[errores["pred"] != errores["real"]][
        ["fecha", "real", "pred",
         "gdelt_n_eventos", "gdelt_goldstein_min",
         "ukmto_n_incidentes", "maritime_severidad_max"]
    ]
    print(f"  Total errores: {len(errores)}")
    if len(errores):
        print(errores.to_string(index=False))

    # ── Tabla resumen comparativa ─────────────────────────────────────────────
    print("\n── Comparacion de variantes (val) ─────────────────────")
    print(f"{'Modelo':<20} {'F1-Macro':>9} {'F1-ALTO':>8} {'Kappa':>7}")
    print("-" * 47)
    print(f"{'GaussianNB':<20} {met_gnb_val['f1_macro']:>9.3f} {met_gnb_val['f1_alto']:>8.3f} {met_gnb_val['kappa']:>7.3f}")
    print(f"{'ComplementNB (*)':<20} {met_cnb_val['f1_macro']:>9.3f} {met_cnb_val['f1_alto']:>8.3f} {met_cnb_val['kappa']:>7.3f}")

    import joblib
    joblib.dump(cnb, MDIR / "naive_bayes.pkl")
    joblib.dump(mms, MDIR / "naive_bayes_scaler.pkl")
    print(f"\nModelo final guardado: models/naive_bayes.pkl (ComplementNB)")

    return cnb, data


if __name__ == "__main__":
    main()
