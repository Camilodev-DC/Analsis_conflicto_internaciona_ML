import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import OPENSKY_CLIENT_ID, OPENSKY_CLIENT_SECRET, HORMUZ_BBOX, GOLFO_BBOX

load_dotenv()

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL   = "https://opensky-network.org/api"

def get_token():
    r = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     OPENSKY_CLIENT_ID,
        "client_secret": OPENSKY_CLIENT_SECRET,
    })
    if r.status_code == 200:
        print("Token OpenSky obtenido.")
        return r.json()["access_token"]
    print(f"Error auth OpenSky {r.status_code}: {r.text}")
    return None

def get_flights_bbox(token, bbox, begin_ts, end_ts):
    headers = {"Authorization": f"Bearer {token}"}
    params  = {
        "lamin": bbox["lat_min"], "lamax": bbox["lat_max"],
        "lomin": bbox["lon_min"], "lomax": bbox["lon_max"],
        "begin": int(begin_ts),   "end":   int(end_ts)
    }
    r = requests.get(f"{API_URL}/states/all", headers=headers, params=params)
    if r.status_code == 200:
        states = r.json().get("states", []) or []
        return states
    print(f"Error {r.status_code}: {r.text}")
    return []

if __name__ == "__main__":
    token = get_token()
    if not token:
        exit(1)

    # últimas 1 hora sobre el Estrecho
    now   = datetime.utcnow()
    begin = now - timedelta(hours=1)

    print(f"Consultando vuelos en Hormuz: {begin.strftime('%H:%M')} - {now.strftime('%H:%M')} UTC")
    states = get_flights_bbox(token, HORMUZ_BBOX, begin.timestamp(), now.timestamp())

    cols = ["icao24","callsign","origin_country","time_position","last_contact",
            "longitude","latitude","baro_altitude","on_ground","velocity",
            "true_track","vertical_rate","sensors","geo_altitude","squawk",
            "spi","position_source"]

    if states:
        df = pd.DataFrame(states, columns=cols[:len(states[0])])
        print(f"Vuelos detectados en Hormuz: {len(df)}")
        print(df[["callsign","origin_country","latitude","longitude","baro_altitude","velocity"]].head(10).to_string())
        df.to_csv("../../data/raw/raw_opensky_hormuz.csv", index=False)
        print("Guardado: data/raw/raw_opensky_hormuz.csv")
    else:
        print("Sin vuelos detectados en este momento (normal fuera de horas pico).")
