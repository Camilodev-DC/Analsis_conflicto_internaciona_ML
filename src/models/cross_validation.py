"""
Cross-validation temporal para todos los modelos.
Usa TimeSeriesSplit — unica CV valida para series temporales.
CV aleatoria mezcla pasado y futuro; TimeSeriesSplit siempre entrena
en pasado y evalua en futuro, igual que en produccion real.

5 folds, test_size=60 dias cada uno.
Nota: fold 5 (Nov-Dic) tiene 0 dias ALTO por ausencia de GDELT —
se reporta pero no se incluye en el promedio de F1-ALTO.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from src.models.preparar_dataset import preparar

FIGS = Path(__file__).parent.parent.parent / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)

N_SPLITS   = 5
TEST_SIZE  = 60   # dias por fold
CLASES     = ["BAJO", "MEDIO", "ALTO"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def construir_features_compañero(df):
    df = df.copy()
    df["indice_letalidad"] = df["acled_region_fatalities"] / (df["acled_region_n_eventos"] + 1)
    df["letalidad_ma7"]    = df["indice_letalidad"].rolling(window=7).mean()
    df["shock_letalidad"]  = df["indice_letalidad"].shift(1) / (df["letalidad_ma7"].shift(1) + 1e-6)
    for col in ["acled_region_fatalities", "gdelt_n_eventos", "shock_letalidad"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    return df


def metricas_fold(y_true, y_pred):
    f1m = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1a = f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    kap = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return f1m, f1a, kap


def resumen(nombre, resultados):
    """resultados: lista de (fold, f1m, f1a, kap, n_alto_test)"""
    f1ms = [r[1] for r in resultados]
    # excluir folds sin ALTO para el promedio de F1-ALTO
    f1as = [r[2] for r in resultados if r[4] > 0]
    kaps = [r[3] for r in resultados]
    print(f"\n  {nombre}")
    print(f"  {'Fold':<6} {'F1-Macro':>9} {'F1-ALTO':>9} {'Kappa':>8} {'#ALTO':>6}")
    for fold, f1m, f1a, kap, n_alto in resultados:
        marca = " *" if n_alto == 0 else ""
        print(f"  {fold:<6} {f1m:>9.3f} {f1a:>9.3f} {kap:>8.3f} {n_alto:>6}{marca}")
    print(f"  {'Media':<6} {np.mean(f1ms):>9.3f} "
          f"{np.mean(f1as) if f1as else float('nan'):>9.3f} "
          f"{np.mean(kaps):>8.3f}")
    print(f"  {'Std':<6} {np.std(f1ms):>9.3f} "
          f"{np.std(f1as) if f1as else float('nan'):>9.3f} "
          f"{np.std(kaps):>8.3f}")
    return {
        "nombre":   nombre,
        "f1m_mean": np.mean(f1ms),
        "f1m_std":  np.std(f1ms),
        "f1a_mean": np.mean(f1as) if f1as else float("nan"),
        "f1a_std":  np.std(f1as)  if f1as else float("nan"),
        "kap_mean": np.mean(kaps),
        "kap_std":  np.std(kaps),
    }


# ── CV por modelo ─────────────────────────────────────────────────────────────

def cv_gaussian_nb(X_full, y_full, tscv):
    resultados = []
    for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
        Xtr, ytr = X_full[tr], y_full[tr]
        Xte, yte = X_full[te], y_full[te]
        # MinMaxScaler: fit solo en train
        mms = MinMaxScaler()
        Xtr_mm = mms.fit_transform(Xtr)
        Xte_mm = mms.transform(Xte)
        # Prior ajustado
        clases, counts = np.unique(ytr, return_counts=True)
        w = 1.0 / counts; w /= w.sum()
        prior = np.zeros(3);
        for c, p in zip(clases, w): prior[c] = p
        gnb = GaussianNB(priors=prior)
        gnb.fit(Xtr_mm, ytr)
        yp = gnb.predict(Xte_mm)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("GaussianNB", resultados)


def cv_complement_nb(X_full, y_full, tscv):
    resultados = []
    for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
        Xtr, ytr = X_full[tr], y_full[tr]
        Xte, yte = X_full[te], y_full[te]
        mms = MinMaxScaler()
        Xtr_mm = mms.fit_transform(Xtr)
        Xte_mm = mms.transform(Xte)
        cnb = ComplementNB()
        cnb.fit(Xtr_mm, ytr)
        yp = cnb.predict(Xte_mm)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("ComplementNB", resultados)


def cv_knn(X_full, y_full, tscv):
    resultados = []
    for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
        Xtr, ytr = X_full[tr], y_full[tr]
        Xte, yte = X_full[te], y_full[te]
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)
        knn = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="manhattan")
        knn.fit(Xtr_sc, ytr)
        yp = knn.predict(Xte_sc)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("KNN k=3 Manhattan", resultados)


def cv_lr(X_full, y_full, tscv):
    resultados = []
    for fold, (tr, te) in enumerate(tscv.split(X_full), 1):
        Xtr, ytr = X_full[tr], y_full[tr]
        Xte, yte = X_full[te], y_full[te]
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)
        lr = LogisticRegression(penalty="l1", C=0.5, solver="saga",
                                class_weight="balanced", max_iter=3000, random_state=42)
        lr.fit(Xtr_sc, ytr)
        yp = lr.predict(Xte_sc)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("LR L1 C=0.5", resultados)


def cv_knn_turbo(df_comp, y_full, tscv):
    feats = [c for c in df_comp.columns if c not in ["fecha", "nivel_riesgo", "target"]]
    X_comp = df_comp[feats].values
    # Alinear indices con y_full (df_comp puede tener menos filas por dropna)
    resultados = []
    idx_comp = df_comp.index.values
    for fold, (tr, te) in enumerate(tscv.split(X_comp), 1):
        Xtr, ytr = X_comp[tr], y_full[idx_comp[tr]]
        Xte, yte = X_comp[te], y_full[idx_comp[te]]
        # SelectKBest solo en train
        sel = SelectKBest(f_classif, k=15)
        Xtr_s = sel.fit_transform(Xtr, ytr)
        Xte_s = sel.transform(Xte)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr_s)
        Xte_sc = sc.transform(Xte_s)
        pca = PCA(n_components=min(10, Xtr_sc.shape[1]))
        Xtr_p = pca.fit_transform(Xtr_sc)
        Xte_p = pca.transform(Xte_sc)
        k_nb  = min(2, (ytr == 2).sum() - 1) if (ytr == 2).sum() > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=max(1, k_nb))
        try:
            Xtr_r, ytr_r = smote.fit_resample(Xtr_p, ytr)
        except Exception:
            Xtr_r, ytr_r = Xtr_p, ytr
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
        knn.fit(Xtr_r, ytr_r)
        yp = knn.predict(Xte_p)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("KNN Turbo", resultados)


def cv_rf(df_comp, y_full, tscv):
    feats = [c for c in df_comp.columns if c not in ["fecha", "nivel_riesgo", "target"]]
    X_comp = df_comp[feats].values
    idx_comp = df_comp.index.values
    resultados = []
    for fold, (tr, te) in enumerate(tscv.split(X_comp), 1):
        Xtr, ytr = X_comp[tr], y_full[idx_comp[tr]]
        Xte, yte = X_comp[te], y_full[idx_comp[te]]
        sel = SelectKBest(f_classif, k=15)
        Xtr_s = sel.fit_transform(Xtr, ytr)
        Xte_s = sel.transform(Xte)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr_s)
        Xte_sc = sc.transform(Xte_s)
        try:
            smt = SMOTETomek(random_state=42)
            Xtr_r, ytr_r = smt.fit_resample(Xtr_sc, ytr)
        except Exception:
            Xtr_r, ytr_r = Xtr_sc, ytr
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight={0:1, 1:1.5, 2:3},
                                    random_state=42, n_jobs=-1)
        rf.fit(Xtr_r, ytr_r)
        yp = rf.predict(Xte_sc)
        f1m, f1a, kap = metricas_fold(yte, yp)
        resultados.append((fold, f1m, f1a, kap, int((yte == 2).sum())))
    return resumen("RF Robusto", resultados)


# ── Figura comparativa ────────────────────────────────────────────────────────

def plot_cv_comparacion(resumenes, ruta):
    nombres = [r["nombre"] for r in resumenes]
    f1ms    = [r["f1m_mean"] for r in resumenes]
    f1ms_e  = [r["f1m_std"]  for r in resumenes]
    f1as    = [r["f1a_mean"] for r in resumenes]
    f1as_e  = [r["f1a_std"]  for r in resumenes]
    kaps    = [r["kap_mean"] for r in resumenes]

    x   = np.arange(len(nombres))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    colores = ["steelblue", "darkorange", "crimson"]
    etiquetas = ["F1-Macro", "F1-ALTO", "Kappa"]
    datos = [(f1ms, f1ms_e), (f1as, [0]*len(f1as)), (kaps, [0]*len(kaps))]

    for i, (vals, errs) in enumerate(datos):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, yerr=errs,
                      label=etiquetas[i], color=colores[i],
                      alpha=0.85, capsize=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Cross-Validation Temporal (TimeSeriesSplit 5 folds)\nBarras = media ± std", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"\n  Figura guardada: {ruta.name}")


def plot_cv_folds(todos_resultados, ruta):
    """Linea de F1-Macro por fold para cada modelo."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colores = ["steelblue", "darkorange", "crimson", "seagreen", "purple", "brown"]
    for i, (nombre, resultados) in enumerate(todos_resultados.items()):
        folds = [r[0] for r in resultados]
        f1ms  = [r[1] for r in resultados]
        ax.plot(folds, f1ms, "o-", label=nombre, color=colores[i % len(colores)], lw=1.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("F1-Macro")
    ax.set_title("F1-Macro por fold temporal — todos los modelos")
    ax.set_xticks([1,2,3,4,5])
    ax.axvline(4.5, color="gray", linestyle="--", lw=1,
               label="Fold 5: sin días ALTO (GDELT ausente)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CROSS-VALIDATION TEMPORAL — TimeSeriesSplit 5 folds")
    print("  test_size=60 dias | Solo folds 1-4 con dias ALTO")
    print("=" * 65)

    data     = preparar(verbose=False)
    X_full   = np.vstack([data["X_train_sc"], data["X_val_sc"], data["X_test_sc"]])
    y_full   = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    df_base  = data["df_full"].copy()

    # Dataset extendido del compañero
    df_comp  = construir_features_compañero(df_base).dropna().reset_index(drop=True)
    y_comp   = df_comp["target"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)

    print("\n* = fold excluido del promedio F1-ALTO (0 dias ALTO en test)\n")

    resumenes = []
    todos_raw = {}   # para la figura por folds

    # Tus modelos
    for fn, nombre in [
        (lambda: cv_gaussian_nb(X_full, y_full, tscv),   "GaussianNB"),
        (lambda: cv_complement_nb(X_full, y_full, tscv), "ComplementNB"),
        (lambda: cv_knn(X_full, y_full, tscv),           "KNN k=3 Manhattan"),
        (lambda: cv_lr(X_full, y_full, tscv),            "LR L1 C=0.5"),
    ]:
        r = fn()
        resumenes.append(r)

    # Modelos del compañero (usan features adicionales)
    tscv_comp = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    for fn, nombre in [
        (lambda: cv_knn_turbo(df_comp, y_comp, tscv_comp), "KNN Turbo"),
        (lambda: cv_rf(df_comp, y_comp, tscv_comp),        "RF Robusto"),
    ]:
        r = fn()
        resumenes.append(r)

    # ── Tabla final ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL — medias sobre folds 1-5")
    print("  (F1-ALTO promediado solo sobre folds con dias ALTO)")
    print("=" * 65)
    print(f"\n  {'Modelo':<22} {'F1-Macro':>9} {'±':>4} {'F1-ALTO':>9} {'Kappa':>8}")
    print("  " + "-" * 57)
    for r in sorted(resumenes, key=lambda x: -x["f1m_mean"]):
        f1a_str = f"{r['f1a_mean']:.3f}" if not np.isnan(r['f1a_mean']) else "  NaN"
        print(f"  {r['nombre']:<22} {r['f1m_mean']:>9.3f} {r['f1m_std']:>4.3f} "
              f"{f1a_str:>9} {r['kap_mean']:>8.3f}")

    # ── Figuras ───────────────────────────────────────────────────────────────
    plot_cv_comparacion(resumenes, FIGS / "cv_comparacion.png")

    return resumenes


if __name__ == "__main__":
    main()
