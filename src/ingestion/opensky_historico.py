import requests
import pandas as pd
from datetime import datetime, timezone
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENSKY_CLIENT_ID, OPENSKY_CLIENT_SECRET, HORMUZ_BBOX, GOLFO_BBOX

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL   = "https://opensky-network.org/api"

# ─── FECHAS CLAVE DEL CONFLICTO ───────────────────────────────────────────────
FECHAS_CLAVE = [
    # Días de escalada máxima
    ("2024-04-13", "ataque_iran_israel_dia1"),
    ("2024-04-14", "ataque_iran_israel_dia2"),
    ("2024-04-15", "post_ataque"),
    ("2024-01-28", "ataque_torre22_jordan"),
    ("2024-10-01", "respuesta_israel_iran"),
    ("2024-10-02", "post_respuesta"),
    # Días de control (tensión baja)
    ("2024-03-01", "control_marzo"),
    ("2024-06-15", "control_junio"),
    ("2024-08-10", "control_agosto"),
]

BBOXES = {
    "hormuz": HORMUZ_BBOX,
    "golfo":  GOLFO_BBOX,
}


def get_token():
    r = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     OPENSKY_CLIENT_ID,
        "client_secret": OPENSKY_CLIENT_SECRET,
    })
    if r.status_code == 200:
        print("Token OpenSky OK")
        return r.json()["access_token"]
    print(f"Error auth: {r.status_code} {r.text}")
    return None


def fecha_a_ts(fecha_str, hora="12:00:00"):
    dt = datetime.strptime(f"{fecha_str} {hora}", "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def get_flights(token, bbox, begin_ts, end_ts, retries=3):
    headers = {"Authorization": f"Bearer {token}"}
    params  = {
        "lamin": bbox["lat_min"], "lamax": bbox["lat_max"],
        "lomin": bbox["lon_min"], "lomax": bbox["lon_max"],
        "begin": begin_ts,        "end":   end_ts
    }
    for attempt in range(retries):
        r = requests.get(f"{API_URL}/states/all", headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            return r.json().get("states", []) or []
        elif r.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit — esperando {wait}s...")
            time.sleep(wait)
        else:
            print(f"  Error {r.status_code}: {r.text[:100]}")
            break
    return []


COLS = ["icao24", "callsign", "origin_country", "time_position", "last_contact",
        "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
        "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
        "spi", "position_source"]


if __name__ == "__main__":
    token = get_token()
    if not token:
        exit(1)

    all_records = []

    for fecha_str, label in FECHAS_CLAVE:
        begin_ts = fecha_a_ts(fecha_str, "06:00:00")
        end_ts   = fecha_a_ts(fecha_str, "18:00:00")

        for region_name, bbox in BBOXES.items():
            print(f"[{fecha_str}] [{label}] [{region_name}] consultando...")
            states = get_flights(token, bbox, begin_ts, end_ts)

            if states:
                df = pd.DataFrame(states, columns=COLS[:len(states[0])])
                df["fecha"]      = fecha_str
                df["label"]      = label
                df["region"]     = region_name
                all_records.append(df)
                print(f"  >> {len(df)} vuelos detectados")
            else:
                print(f"  >> 0 vuelos (posible límite de créditos o sin datos)")

            time.sleep(2)

    if all_records:
        df_final = pd.concat(all_records, ignore_index=True)
        df_final["baro_altitude"] = pd.to_numeric(df_final["baro_altitude"], errors="coerce")
        df_final["velocity"]      = pd.to_numeric(df_final["velocity"],      errors="coerce")
        df_final["on_ground"]     = df_final["on_ground"].astype(str).str.lower() == "true"

        print(f"\nTotal vuelos descargados: {len(df_final)}")
        print(f"Fechas cubiertas: {df_final['fecha'].unique()}")
        print(f"\nVuelos por fecha y región:")
        print(df_final.groupby(["fecha","region"])["icao24"].count().to_string())

        df_final.to_csv("../../data/raw/raw_opensky_historico.csv", index=False)
        print("\nGuardado: data/raw/raw_opensky_historico.csv")
    else:
        print("Sin datos obtenidos — verifica créditos de OpenSky.")
