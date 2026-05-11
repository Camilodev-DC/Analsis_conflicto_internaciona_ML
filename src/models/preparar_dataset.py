"""
Preparacion del dataset para modelado ML.
Aplica limpieza, transformaciones y splits temporales 60/20/20.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
MDIR = ROOT / "models"
MDIR.mkdir(exist_ok=True)

# ── Features a eliminar (varianza cero, multicolinealidad, OpenSky) ────────────
ELIMINAR = [
    "opensky_n_grounded",       # varianza cero
    "ukmto_n_hijack",           # varianza cero
    "ukmto_n_suspicious",       # 99.7% ceros
    "opensky_n_airborne",       # identica a opensky_n_vuelos
    "opensky_n_vuelos",         # 97.5% ceros - inutilizable
    "opensky_alt_media",        # 97.5% ceros
    "opensky_vel_media",        # 97.5% ceros
    "gdelt_n_articles",         # r=0.98 con gdelt_n_eventos
    "gdelt_n_mentions",         # r=0.97 con gdelt_n_eventos
    "maritime_severidad_sum",   # r=0.96 con maritime_severidad_max
    "acled_hormuz_fatalities",  # r=1.00 con acled_hormuz_n_violentos
    "dia_del_año",              # r=0.997 con mes
]

# ── Features a transformar con log1p (skewness > 3) ───────────────────────────
LOG1P = [
    "gdelt_n_conflicto",
    "gdelt_n_eventos",
    "ukmto_n_attacks",
    "gdelt_n_conflicto_ma7",
    "acled_region_fatalities",
]

# ── Features a binarizar (>99% ceros) ─────────────────────────────────────────
BINARIZAR = [
    "acled_hormuz_n_violentos",
    "maritime_n_hormuz",
]


def preparar(verbose=True):
    df = pd.read_csv(PROC / "dataset_integrado.csv", parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    if verbose:
        print(f"Dataset cargado: {df.shape}")
        print(f"Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")

    # ── Eliminar features problemáticas ──────────────────────────────────────
    cols_a_eliminar = [c for c in ELIMINAR if c in df.columns]
    df = df.drop(columns=cols_a_eliminar)
    if verbose:
        print(f"Eliminadas {len(cols_a_eliminar)} columnas redundantes/vacias")

    # ── Transformaciones log1p ────────────────────────────────────────────────
    for col in LOG1P:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # ── Binarizar ─────────────────────────────────────────────────────────────
    for col in BINARIZAR:
        if col in df.columns:
            df[col] = (df[col] > 0).astype(int)

    # ── Target encoding ───────────────────────────────────────────────────────
    label_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}
    df["target"] = df["nivel_riesgo"].map(label_map)

    # ── Split temporal 60 / 20 / 20 ──────────────────────────────────────────
    n = len(df)
    n_train = int(n * 0.60)   # 220 dias  Ene–Jul
    n_val   = int(n * 0.20)   # 73 dias   Ago–Sep
    # n_test  = resto          # 73 dias   Oct–Dic

    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train + n_val].copy()
    df_test  = df.iloc[n_train + n_val:].copy()

    if verbose:
        print(f"\nSplit temporal:")
        print(f"  Train: {len(df_train)} dias | {df_train['fecha'].min().date()} → {df_train['fecha'].max().date()}")
        print(f"  Val:   {len(df_val)}   dias | {df_val['fecha'].min().date()} → {df_val['fecha'].max().date()}")
        print(f"  Test:  {len(df_test)}  dias | {df_test['fecha'].min().date()} → {df_test['fecha'].max().date()}")
        print(f"\n  Distribucion train:")
        print(f"  {df_train['nivel_riesgo'].value_counts().to_string()}")
        print(f"\n  Distribucion val:")
        print(f"  {df_val['nivel_riesgo'].value_counts().to_string()}")
        print(f"\n  Distribucion test:")
        print(f"  {df_test['nivel_riesgo'].value_counts().to_string()}")

    # ── Separar X / y ─────────────────────────────────────────────────────────
    drop_cols = ["fecha", "nivel_riesgo", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val["target"].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test["target"].values

    # ── Escalar (fit solo en train) ───────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Guardar scaler y nombres de features
    joblib.dump(scaler, MDIR / "scaler.pkl")
    joblib.dump(feature_cols, MDIR / "feature_cols.pkl")

    if verbose:
        print(f"\nFeatures finales ({len(feature_cols)}):")
        for f in feature_cols:
            print(f"  - {f}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "X_train_sc": X_train_sc,
        "X_val_sc":   X_val_sc,
        "X_test_sc":  X_test_sc,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "label_map": label_map,
        "df_train": df_train,
        "df_val":   df_val,
        "df_test":  df_test,
        "df_full":  df,
    }


if __name__ == "__main__":
    data = preparar(verbose=True)
    print("\nDataset listo para modelado.")
