"""
Pipeline de integración de datos multi-fuente → dataset_integrado.csv
Fuentes: ACLED (región) + GDELT + OpenSky + Maritime incidents manuales
Unidad de análisis: día en región Hormuz/Mar Rojo
Target: nivel_riesgo = BAJO / MEDIO / ALTO
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path(__file__).parent.parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Parámetros para nivel de riesgo ─────────────────────────────────────────
# ALTO: ≥3 eventos violentos en la región OR fatalities>0 OR severidad_max≥4
# MEDIO: 1-2 eventos violentos OR severidad 2-3
# BAJO: sin eventos violentos directos

VIOLENT_TYPES = {
    "Explosions/Remote violence", "Battles", "Violence against civilians"
}


# ── 1. ACLED — agregar por día (dos niveles) ─────────────────────────────────
def build_acled_daily():
    # Región amplia (Yemen, Irak, Irán, EEUU, etc.) — señal geopolítica general
    df_region = pd.read_csv(RAW / "raw_acled_region.csv", parse_dates=["event_date"])
    df_region["fecha"] = df_region["event_date"].dt.date.astype(str)
    df_region["is_violent"] = df_region["event_type"].isin(VIOLENT_TYPES).astype(int)

    agg_region = df_region.groupby("fecha").agg(
        acled_region_n_eventos   = ("event_type", "count"),
        acled_region_n_violentos = ("is_violent", "sum"),
        acled_region_fatalities  = ("fatalities", "sum"),
        acled_region_n_explosions= ("event_type", lambda x: (x == "Explosions/Remote violence").sum()),
    ).reset_index()

    # Hormuz específico (solo Iran + UAE + Oman bbox) — señal directa
    df_hormuz = pd.read_csv(RAW / "raw_acled_hormuz.csv", parse_dates=["event_date"])
    df_hormuz["fecha"] = df_hormuz["event_date"].dt.date.astype(str)
    df_hormuz["is_violent"] = df_hormuz["event_type"].isin(VIOLENT_TYPES).astype(int)

    agg_hormuz = df_hormuz.groupby("fecha").agg(
        acled_hormuz_n_eventos   = ("event_type", "count"),
        acled_hormuz_n_violentos = ("is_violent", "sum"),
        acled_hormuz_fatalities  = ("fatalities", "sum"),
    ).reset_index()

    agg = agg_region.merge(agg_hormuz, on="fecha", how="outer")
    print(f"ACLED diario: {len(agg)} dias")
    return agg


# ── 2. GDELT — agregar por día ───────────────────────────────────────────────
def build_gdelt_daily():
    df = pd.read_csv(RAW / "raw_gdelt_2024_hormuz.csv",
                     encoding="utf-8", encoding_errors="replace",
                     low_memory=False)

    # Day es formato YYYYMMDD int
    df["fecha"] = pd.to_datetime(df["Day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["fecha"] = df["fecha"].dt.date.astype(str)

    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumMentions"]    = pd.to_numeric(df["NumMentions"],    errors="coerce").fillna(0)
    df["NumArticles"]    = pd.to_numeric(df["NumArticles"],    errors="coerce").fillna(0)
    df["is_conflict"]    = (df["QuadClass"] == 3).astype(int)  # Material Conflict

    agg = df.groupby("fecha").agg(
        gdelt_n_eventos        = ("GlobalEventID", "count"),
        gdelt_goldstein_mean   = ("GoldsteinScale", "mean"),
        gdelt_goldstein_min    = ("GoldsteinScale", "min"),
        gdelt_n_mentions       = ("NumMentions", "sum"),
        gdelt_n_articles       = ("NumArticles", "sum"),
        gdelt_n_conflicto      = ("is_conflict", "sum"),
    ).reset_index()

    print(f"GDELT diario: {len(agg)} días | {agg['fecha'].min()} → {agg['fecha'].max()}")
    return agg


# ── 3. OpenSky — agregar por día ─────────────────────────────────────────────
def build_opensky_daily():
    df = pd.read_csv(RAW / "raw_opensky_historico.csv")
    df["on_ground"] = df["on_ground"].astype(str).str.lower() == "true"

    agg = df.groupby("fecha").agg(
        opensky_n_vuelos     = ("icao24", "count"),
        opensky_n_airborne   = ("on_ground", lambda x: (~x).sum()),
        opensky_n_grounded   = ("on_ground", "sum"),
        opensky_alt_media    = ("baro_altitude", "mean"),
        opensky_vel_media    = ("velocity", "mean"),
    ).reset_index()

    print(f"OpenSky diario: {len(agg)} días | {agg['fecha'].min()} → {agg['fecha'].max()}")
    return agg


# ── 4. Maritime incidents — ya agregado ──────────────────────────────────────
def build_maritime_daily():
    df = pd.read_csv(PROC / "maritime_agregado_diario.csv", parse_dates=["fecha"])
    df["fecha"] = df["fecha"].dt.date.astype(str)
    print(f"Maritime diario: {len(df)} dias")
    return df


# ── 5. UKMTO 2024 archivo — agregado ─────────────────────────────────────────
def build_ukmto_daily():
    df = pd.read_csv(PROC / "ukmto_2024_diario.csv")
    print(f"UKMTO 2024 diario: {len(df)} dias")
    return df


# ── 5. JOIN por fecha ─────────────────────────────────────────────────────────
def build_base_dates():
    """Genera el esqueleto diario 2024-01-01 → 2024-12-31"""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({"fecha": dates.strftime("%Y-%m-%d")})


def join_all():
    base     = build_base_dates()
    acled    = build_acled_daily()
    gdelt    = build_gdelt_daily()
    opensky  = build_opensky_daily()
    maritime = build_maritime_daily()
    ukmto    = build_ukmto_daily()

    df = base.merge(acled,    on="fecha", how="left")
    df = df.merge(gdelt,      on="fecha", how="left")
    df = df.merge(opensky,    on="fecha", how="left")
    df = df.merge(maritime,   on="fecha", how="left")
    df = df.merge(ukmto,      on="fecha", how="left")

    # Rellenar NaN con 0 para features numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"\nDataset integrado: {len(df)} días, {df.shape[1]} columnas")
    return df


# ── 6. Target variable — nivel_riesgo ────────────────────────────────────────
def assign_nivel_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reglas de negocio para nivel_riesgo diario en estrecho de Hormuz.

    ALTO:  ≥1 evento violento DIRECTO en Hormuz (acled_hormuz_n_violentos>0)
           OR incidente marítimo de severidad ≥4 ese día
           OR acled_hormuz_fatalities > 0
    MEDIO: incidente marítimo de severidad 2-3
           OR >30 explosiones en región (Yemen/Iraq/Irán — presión indirecta)
           OR gdelt_n_conflicto alto (>10 artículos de conflicto ese día)
    BAJO:  resto
    """
    def classify(row):
        # ALTO: incidente físico directo confirmado en Hormuz / Mar Rojo
        if (row.get("acled_hormuz_n_violentos", 0) > 0
                or row.get("acled_hormuz_fatalities", 0) > 0
                or row.get("maritime_severidad_max", 0) >= 4
                or row.get("ukmto_n_attacks", 0) >= 2):
            return "ALTO"

        # MEDIO: incidente documentado o escalada regional significativa
        if (row.get("ukmto_n_attacks", 0) >= 1
                or row.get("ukmto_n_incidentes", 0) >= 1
                or row.get("maritime_severidad_max", 0) >= 2
                or row.get("maritime_n_incidentes", 0) >= 1
                or row.get("acled_region_n_explosions", 0) > 35
                or row.get("gdelt_n_conflicto", 0) > 10):
            return "MEDIO"

        return "BAJO"

    df["nivel_riesgo"] = df.apply(classify, axis=1)
    return df


# ── 7. Feature engineering adicional ─────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"])
    df["dia_semana"]   = df["fecha_dt"].dt.dayofweek      # 0=Mon
    df["mes"]          = df["fecha_dt"].dt.month
    df["dia_del_año"]  = df["fecha_dt"].dt.dayofyear

    # Medias móviles de 7 días (lag para evitar data leakage)
    df = df.sort_values("fecha_dt")
    for col in ["acled_hormuz_n_violentos", "acled_region_n_explosions", "gdelt_n_conflicto",
                "maritime_severidad_sum", "maritime_n_incidentes", "ukmto_n_attacks"]:
        if col in df.columns:
            df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean().shift(1).fillna(0)

    df = df.drop(columns=["fecha_dt"])
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = join_all()
    df = assign_nivel_riesgo(df)
    df = feature_engineering(df)

    print(f"\nDistribución nivel_riesgo:")
    print(df["nivel_riesgo"].value_counts().to_string())

    print(f"\nColumnas finales ({len(df.columns)}):")
    print([c for c in df.columns])

    out = PROC / "dataset_integrado.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado: {out}")

    # Guardar también versión solo con los días que tienen datos de al menos 2 fuentes
    mask = ((df["acled_region_n_eventos"] > 0) | (df["gdelt_n_eventos"] > 0)
            | (df["maritime_n_incidentes"] > 0) | (df["ukmto_n_incidentes"] > 0))
    df_con_datos = df[mask]
    out2 = PROC / "dataset_integrado_con_datos.csv"
    df_con_datos.to_csv(out2, index=False)
    print(f"Versión con datos: {len(df_con_datos)} días → {out2}")
