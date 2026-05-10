import requests
import pandas as pd
import zipfile
import io
import time
import os
from datetime import datetime, timedelta

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
# Meses clave del conflicto Irán-Israel 2024:
# Abril 2024  → ataque de Irán a Israel (13-14 abril)
# Octubre 2024 → respuesta de Israel a Irán
FECHA_INICIO = datetime(2024, 4, 1)
FECHA_FIN    = datetime(2024, 10, 31)

OUTPUT_FILE  = "raw_gdelt_2024_hormuz.csv"
SLEEP_SEC    = 0.3   # pausa entre descargas para no saturar

KEYWORDS_GEO = ["Iran", "Israel", "Hormuz", "Persian Gulf", "Gaza", "Lebanon"]
PAISES_COD   = ["IRN", "ISR", "UAE", "OMN", "US", "SAU"]

COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_Lat", "Actor2Geo_Long",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_Lat", "ActionGeo_Long",
    "DATEADDED", "SOURCEURL"
]


def generar_urls_diarias(fecha_inicio, fecha_fin):
    """Genera URLs de archivos GDELT 2.0 — un archivo por día (00:00:00 UTC)"""
    urls = []
    fecha = fecha_inicio
    while fecha <= fecha_fin:
        stamp = fecha.strftime("%Y%m%d") + "000000"
        url = f"http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
        urls.append((fecha.strftime("%Y-%m-%d"), url))
        fecha += timedelta(days=1)
    return urls


def download_and_filter(fecha_str, url):
    """Descarga un archivo ZIP de GDELT y filtra eventos relevantes"""
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            return None  # archivo no existe para esa fecha
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(
                    f, sep="\t", header=None,
                    low_memory=False,
                    encoding="utf-8", encoding_errors="replace"
                )

        n_cols = min(len(COLS), df.shape[1])
        df = df.iloc[:, :n_cols].copy()
        df.columns = COLS[:n_cols]

        # filtrar por país o mención geográfica
        mask = pd.Series([False] * len(df), index=df.index)
        for col in ["Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode"]:
            if col in df.columns:
                mask |= df[col].isin(PAISES_COD)
        for col in ["Actor1Geo_FullName", "Actor2Geo_FullName", "ActionGeo_FullName"]:
            if col in df.columns:
                for kw in KEYWORDS_GEO:
                    mask |= df[col].astype(str).str.contains(kw, case=False, na=False)

        df_fil = df[mask].copy()
        return df_fil

    except Exception as e:
        print(f"  Error {fecha_str}: {e}")
        return None


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    urls = generar_urls_diarias(FECHA_INICIO, FECHA_FIN)
    print(f"Fechas a descargar: {len(urls)} dias ({FECHA_INICIO.date()} a {FECHA_FIN.date()})")
    print(f"Archivo de salida: {OUTPUT_FILE}\n")

    # si ya existe, retomar desde donde quedó
    dias_ya_descargados = set()
    if os.path.exists(OUTPUT_FILE):
        df_existente = pd.read_csv(OUTPUT_FILE, low_memory=False)
        if "Day" in df_existente.columns:
            dias_ya_descargados = set(df_existente["Day"].astype(str).unique())
        print(f"Retomando — dias ya descargados: {len(dias_ya_descargados)}")

    frames = []
    total_eventos = 0

    for i, (fecha_str, url) in enumerate(urls):
        dia_key = fecha_str.replace("-", "")
        if dia_key in dias_ya_descargados:
            print(f"[{i+1}/{len(urls)}] {fecha_str} — ya descargado, saltando")
            continue

        print(f"[{i+1}/{len(urls)}] {fecha_str} descargando...", end=" ")
        df_fil = download_and_filter(fecha_str, url)

        if df_fil is not None and len(df_fil) > 0:
            frames.append(df_fil)
            total_eventos += len(df_fil)
            print(f"{len(df_fil)} eventos relevantes (total acum: {total_eventos})")
        else:
            print("0 eventos o no disponible")

        # guardar cada 10 dias para no perder progreso
        if len(frames) > 0 and (i + 1) % 10 == 0:
            df_parcial = pd.concat(frames, ignore_index=True)
            if os.path.exists(OUTPUT_FILE):
                df_parcial.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            else:
                df_parcial.to_csv(OUTPUT_FILE, index=False)
            frames = []
            print(f"  >>> Guardado parcial en {OUTPUT_FILE}")

        time.sleep(SLEEP_SEC)

    # guardar el resto
    if frames:
        df_parcial = pd.concat(frames, ignore_index=True)
        if os.path.exists(OUTPUT_FILE):
            df_parcial.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        else:
            df_parcial.to_csv(OUTPUT_FILE, index=False)

    # resumen final
    if os.path.exists(OUTPUT_FILE):
        df_final = pd.read_csv(OUTPUT_FILE, low_memory=False)
        df_final = df_final.drop_duplicates(subset=["GlobalEventID"])
        df_final.to_csv(OUTPUT_FILE, index=False)

        print(f"\n{'='*50}")
        print(f"DESCARGA COMPLETA")
        print(f"Total eventos unicos: {len(df_final)}")
        print(f"Periodo: {df_final['Day'].min()} a {df_final['Day'].max()}")
        print(f"Paises mas frecuentes:\n{df_final['ActionGeo_CountryCode'].value_counts().head(8)}")
        gs = pd.to_numeric(df_final["GoldsteinScale"], errors="coerce")
        print(f"Tono promedio (GoldsteinScale): {gs.mean():.2f}")
        print(f"Archivo guardado: {OUTPUT_FILE}")
