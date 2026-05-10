import requests
import pandas as pd
import zipfile
import io
import time
from datetime import datetime, timedelta

# ─── GDELT 2.0 RAW FILES (sin rate limit) ────────────────────────────────────
# Descarga los archivos CSV de eventos cada 15 min directamente
# Columnas GDELT 2.0: http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf

KEYWORDS = ["Hormuz", "Iran", "Israel", "Persian Gulf", "Strait"]

GDELT_MASTER = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# Cabeceras GDELT 2.0 (las primeras más útiles)
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


def get_latest_gdelt_urls():
    """Obtiene URLs de los últimos 3 archivos de eventos GDELT 2.0"""
    print("Obteniendo lista de archivos GDELT más recientes...")
    r = requests.get(GDELT_MASTER, timeout=15)
    r.raise_for_status()

    urls = []
    for line in r.text.strip().split("\n"):
        parts = line.strip().split(" ")
        if len(parts) == 3:
            url = parts[2]
            if "export" in url:  # archivo de eventos (no menciones ni GKG)
                urls.append(url)
    return urls


def download_gdelt_file(url):
    """Descarga y parsea un archivo ZIP de eventos GDELT"""
    print(f"Descargando: {url.split('/')[-1]}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            df = pd.read_csv(f, sep="\t", header=None, low_memory=False, encoding="utf-8", encoding_errors="replace")

    # asignar solo las columnas que existen
    n_cols = min(len(COLS), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = COLS[:n_cols]
    return df


def filter_hormuz(df):
    """Filtra eventos relacionados con el Estrecho de Hormuz / conflicto Irán"""
    mask = pd.Series([False] * len(df))

    # por texto en nombre geográfico
    for col in ["Actor1Geo_FullName", "Actor2Geo_FullName", "ActionGeo_FullName"]:
        if col in df.columns:
            for kw in KEYWORDS:
                mask |= df[col].astype(str).str.contains(kw, case=False, na=False)

    # por código de país (IRN=Irán, ISR=Israel, UAE=Emiratos, OMN=Omán)
    for col in ["Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode"]:
        if col in df.columns:
            mask |= df[col].isin(["IRN", "ISR", "UAE", "OMN", "US"])

    return df[mask].copy()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    urls = get_latest_gdelt_urls()
    print(f"Archivos encontrados: {len(urls)}")

    all_frames = []
    for url in urls:
        try:
            df = download_gdelt_file(url)
            df_filtered = filter_hormuz(df)
            print(f"  >> {len(df)} eventos totales | {len(df_filtered)} relevantes")
            all_frames.append(df_filtered)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}")

    if all_frames:
        df_final = pd.concat(all_frames, ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["GlobalEventID"])

        print(f"\nTotal eventos relevantes: {len(df_final)}")
        print(f"\nTono promedio (GoldsteinScale): {pd.to_numeric(df_final['GoldsteinScale'], errors='coerce').mean():.2f}")
        print(f"Tono promedio (AvgTone): {pd.to_numeric(df_final['AvgTone'], errors='coerce').mean():.2f}")
        print(f"\nPaíses más mencionados:\n{df_final['ActionGeo_CountryCode'].value_counts().head(8)}")
        print(f"\nMuestra:\n{df_final[['Day','Actor1Name','Actor2Name','ActionGeo_FullName','GoldsteinScale','SOURCEURL']].head(5).to_string()}")

        df_final.to_csv("raw_gdelt_eventos.csv", index=False)
        print("\nGuardado: raw_gdelt_eventos.csv")
    else:
        print("No se obtuvieron datos.")
