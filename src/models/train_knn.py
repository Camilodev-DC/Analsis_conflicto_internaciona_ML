"""
KNN Turbo del compañero — corregido con split temporal 60/20/20.
Mejoras aplicadas:
  - SelectKBest fit solo sobre train (sin data leakage)
  - PCA n_components=10 (0.971 varianza, mejor que 5 con 0.871)
  - shock_letalidad calculado con shift(1) para evitar lookahead
  - sin target_lag1 (causa data leakage directo)
  - StandardScaler obligatorio antes de PCA y KNN
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
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
    """Agrega features de ingenieria del compañero con shift correcto."""
    df = df_full.copy()
    df["indice_letalidad"] = df["acled_region_fatalities"] / (df["acled_region_n_eventos"] + 1)
    df["letalidad_ma7"]    = df["indice_letalidad"].rolling(window=7).mean()
    # shift(1): usa solo informacion del dia ANTERIOR, sin lookahead
    df["shock_letalidad"]  = df["indice_letalidad"].shift(1) / (df["letalidad_ma7"].shift(1) + 1e-6)
    for col in ["acled_region_fatalities", "gdelt_n_eventos", "shock_letalidad"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)


def train_knn_turbo():
    print("=" * 60)
    print("  KNN TURBO — Split temporal + PCA=10 + Manhattan")
    print("=" * 60)

    data = preparar(verbose=False)
    df_full = construir_features_compañero(data["df_full"])

    # Split temporal 60/20/20
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
    print(f"Clases val: {dict(pd.Series(y_v).value_counts().sort_index())}")

    # SelectKBest SOLO sobre train
    selector = SelectKBest(f_classif, k=15)
    Xtr_sel = selector.fit_transform(df_tr[feats].values, y_tr)
    Xv_sel  = selector.transform(df_v[feats].values)
    Xt_sel  = selector.transform(df_t[feats].values)
    sel_feats = [feats[i] for i in selector.get_support(indices=True)]
    print(f"\nFeatures seleccionadas: {sel_feats}")

    # StandardScaler obligatorio (KNN mide distancias — magnitudes distintas lo rompen)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr_sel)
    Xv_sc  = scaler.transform(Xv_sel)
    Xt_sc  = scaler.transform(Xt_sel)

    # PCA n=10: captura 97% de varianza vs 87% con n=5
    pca = PCA(n_components=10)
    Xtr_pca = pca.fit_transform(Xtr_sc)
    Xv_pca  = pca.transform(Xv_sc)
    Xt_pca  = pca.transform(Xt_sc)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"PCA 10 componentes: varianza explicada = {var_exp:.3f}")

    # SMOTE sobre train
    smote = SMOTE(random_state=42, k_neighbors=2)
    Xtr_res, y_tr_res = smote.fit_resample(Xtr_pca, y_tr)
    print(f"Clases tras SMOTE: {dict(pd.Series(y_tr_res).value_counts().sort_index())}")

    # KNN con distancia Manhattan (mejor con features asimétricas)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
    knn.fit(Xtr_res, y_tr_res)

    y_pred_v = knn.predict(Xv_pca)
    y_pred_t = knn.predict(Xt_pca)

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

    joblib.dump(knn,      MDIR / "knn_turbo.pkl")
    joblib.dump(scaler,   MDIR / "knn_turbo_scaler.pkl")
    joblib.dump(pca,      MDIR / "knn_turbo_pca.pkl")
    joblib.dump(selector, MDIR / "knn_turbo_selector.pkl")
    print("\nModelos guardados: knn_turbo.pkl + scaler + pca + selector")

    return knn, {"selector": selector, "scaler": scaler, "pca": pca,
                 "feats": feats, "y_val": y_v, "y_pred_val": y_pred_v}


if __name__ == "__main__":
    train_knn_turbo()
