import requests
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIRMS_MAP_KEY, HORMUZ_BBOX, MAR_ROJO_BBOX

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

def check_status(map_key):
    r = requests.get(f"{BASE_URL}/mapserver/mapkey_status/?MAP_KEY={map_key}")
    if r.status_code == 200:
        data = r.json()
        print(f"MAP_KEY OK — transacciones usadas hoy: {data.get('current_transactions', '?')} / 5000")
        return True
    print(f"Error MAP_KEY: {r.status_code}")
    return False

def fetch_firms(map_key, source, bbox, days=5):
    bbox_str = f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"
    url = f"{BASE_URL}/area/csv/{map_key}/{source}/{bbox_str}/{days}"
    print(f"Consultando FIRMS {source} — bbox Hormuz, últimos {days} días...")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text[:200]}")
        return None
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    print(f"Hotspots recibidos: {len(df)}")
    return df

if __name__ == "__main__":
    if not check_status(FIRMS_MAP_KEY):
        exit(1)

    frames = []
    for source in ["MODIS_NRT", "VIIRS_NOAA20_NRT"]:
        df = fetch_firms(FIRMS_MAP_KEY, source, HORMUZ_BBOX, days=5)
        if df is not None and len(df) > 0:
            df["source"] = source
            frames.append(df)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        print(f"\nTotal hotspots Hormuz: {len(df_all)}")
        print(df_all[["latitude","longitude","brightness","frp","acq_date","confidence","source"]].head(10).to_string())
        df_all.to_csv("../../data/raw/raw_firms_hormuz.csv", index=False)
        print("Guardado: data/raw/raw_firms_hormuz.csv")
    else:
        print("Sin hotspots en el bbox de Hormuz en los últimos 5 días.")
