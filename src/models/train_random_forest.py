"""
Random Forest Robusto del compañero — corregido con split temporal 60/20/20.
Mejoras aplicadas:
  - SelectKBest fit solo sobre train (sin data leakage)
  - shock_letalidad con shift(1) correcto
  - sin target_lag1
  - max_depth=8 (mejor F1-ALTO que depth=5 o None)
  - StandardScaler antes de SMOTE-Tomek
  - n_estimators=200
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTETomek
import joblib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.preparar_dataset import preparar

ROOT = Path(__file__).parent.parent.parent
MDIR = ROOT / "models"
FIGS = ROOT / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)

CLASES = ["BAJO", "MEDIO", "ALTO"]


def construir_features_compañero(df_full):
    df = df_full.copy()
    df["indice_letalidad"] = df["acled_region_fatalities"] / (df["acled_region_n_eventos"] + 1)
    df["letalidad_ma7"]    = df["indice_letalidad"].rolling(window=7).mean()
    df["shock_letalidad"]  = df["indice_letalidad"].shift(1) / (df["letalidad_ma7"].shift(1) + 1e-6)
    for col in ["acled_region_fatalities", "gdelt_n_eventos", "shock_letalidad"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)


def train_rf_final():
    print("=" * 60)
    print("  RANDOM FOREST ROBUSTO — Split temporal + depth=8")
    print("=" * 60)

    data = preparar(verbose=False)
    df_full = construir_features_compañero(data["df_full"])

    n = len(df_full)
    n_tr = int(n * 0.60)
    n_v  = int(n * 0.20)
    feats = [c for c in df_full.columns if c not in ["fecha", "nivel_riesgo", "target"]]

    df_tr = df_full.iloc[:n_tr]
    df_v  = df_full.iloc[n_tr:n_tr + n_v]
    df_t  = df_full.iloc[n_tr + n_v:]
    y_tr  = df_tr["target"].values
    y_v   = df_v["target"].values
    y_t   = df_t["target"].values

    print(f"\nTrain: {len(df_tr)} dias | Val: {len(df_v)} | Test: {len(df_t)}")

    # SelectKBest solo sobre train
    selector = SelectKBest(f_classif, k=15)
    Xtr_sel = selector.fit_transform(df_tr[feats].values, y_tr)
    Xv_sel  = selector.transform(df_v[feats].values)
    Xt_sel  = selector.transform(df_t[feats].values)
    sel_feats = [feats[i] for i in selector.get_support(indices=True)]
    print(f"Features seleccionadas: {sel_feats}")

    # StandardScaler
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr_sel)
    Xv_sc  = scaler.transform(Xv_sel)
    Xt_sc  = scaler.transform(Xt_sel)

    # SMOTE-Tomek
    smt = SMOTETomek(random_state=42)
    Xtr_res, y_tr_res = smt.fit_resample(Xtr_sc, y_tr)
    print(f"Clases tras SMOTE-Tomek: {dict(pd.Series(y_tr_res).value_counts().sort_index())}")

    # RF: max_depth=8 maximiza F1-ALTO sin sacrificar F1-Macro
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight={0: 1, 1: 1.5, 2: 3},
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xtr_res, y_tr_res)

    y_pred_v = rf.predict(Xv_sc)
    y_pred_t = rf.predict(Xt_sc)

    f1m = f1_score(y_v, y_pred_v, average="macro", zero_division=0)
    f1a = f1_score(y_v, y_pred_v, labels=[2], average=None, zero_division=0)[0]
    kap = cohen_kappa_score(y_v, y_pred_v)

    print(f"\n── Validación ─────────────────────────────────────────")
    clases_p = sorted(np.unique(np.concatenate([y_v, y_pred_v])))
    print(classification_report(y_v, y_pred_v,
                                 labels=clases_p,
                                 target_names=[CLASES[i] for i in clases_p],
                                 zero_division=0))
    print(f"F1-Macro={f1m:.3f}  F1-ALTO={f1a:.3f}  Kappa={kap:.3f}")

    # Feature importance
    imp = pd.Series(rf.feature_importances_, index=sel_feats).sort_values(ascending=False)
    print(f"\nTop 8 features:")
    print(imp.head(8).to_string())

    joblib.dump(rf,       MDIR / "random_forest.pkl")
    joblib.dump(scaler,   MDIR / "rf_scaler.pkl")
    joblib.dump(selector, MDIR / "rf_selector.pkl")
    print("\nModelo guardado: random_forest.pkl + scaler + selector")

    return rf, {"selector": selector, "scaler": scaler,
                "feats": feats, "y_val": y_v, "y_pred_val": y_pred_v}


if __name__ == "__main__":
    train_rf_final()
