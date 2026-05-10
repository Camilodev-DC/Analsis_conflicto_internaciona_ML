import requests
import pandas as pd
import json

# ─── CREDENCIALES ────────────────────────────────────────────────────────────
EMAIL    = "brayan.hernandez3@est.uexternado.edu.co"
PASSWORD = "BOioqVlCO4HeymTmuMMO^Xsb3"

# ─── PAISES Y AÑO ────────────────────────────────────────────────────────────
PAISES = "Iran|United States|Yemen|Iraq|United Arab Emirates|Oman"
YEAR   = 2024

# ─── OAUTH2: obtener Bearer token ────────────────────────────────────────────
def get_token(email, password):
    r = requests.post(
        "https://acleddata.com/oauth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username":   email,
            "password":   password,
            "grant_type": "password",
            "client_id":  "acled",
            "scope":      "authenticated"
        }
    )
    if r.status_code == 200:
        token = r.json().get("access_token")
        print(f"Token obtenido: {token[:20]}...")
        return token
    else:
        print(f"Error auth {r.status_code}: {r.text}")
        return None


# ─── CONSULTA A LA API ───────────────────────────────────────────────────────
def fetch_acled(token, countries, year, limit=500):
    headers = {"Authorization": f"Bearer {token}"}
    params  = {
        "_format":  "json",
        "country":  countries,
        "year":     year,
        "limit":    limit,
        "page":     1,
        "fields":   "event_id_cnty|event_date|event_type|sub_event_type|actor1|actor2|country|latitude|longitude|fatalities|notes"
    }

    all_data = []
    while True:
        print(f"Consultando pagina {params['page']}...")
        r = requests.get(
            "https://acleddata.com/api/acled/read",
            headers=headers,
            params=params
        )

        if r.status_code != 200:
            print(f"Error HTTP {r.status_code}: {r.text}")
            break

        data   = r.json()
        eventos = data.get("data", [])
        all_data.extend(eventos)
        print(f"  >> {len(eventos)} eventos (total: {len(all_data)})")

        if len(eventos) < limit:
            break
        params["page"] += 1

    return all_data


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    token = get_token(EMAIL, PASSWORD)
    if not token:
        print("No se pudo obtener token. Verifica email y password.")
        exit(1)

    eventos = fetch_acled(token, PAISES, YEAR)

    if eventos:
        df = pd.DataFrame(eventos)
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["latitude"]   = pd.to_numeric(df["latitude"],   errors="coerce")
        df["longitude"]  = pd.to_numeric(df["longitude"],  errors="coerce")
        df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce")

        # filtrar bbox Hormuz
        mask_hormuz = (
            (df["latitude"]  >= 25.0) & (df["latitude"]  <= 27.5) &
            (df["longitude"] >= 55.5) & (df["longitude"] <= 58.0)
        )
        df_hormuz = df[mask_hormuz].copy()

        print(f"\nTotal eventos region: {len(df)}")
        print(f"Eventos en bbox Hormuz: {len(df_hormuz)}")
        print(f"\nTipos de eventos:\n{df['event_type'].value_counts()}")
        print(f"\nMuestra:\n{df[['event_date','country','event_type','fatalities']].head(5).to_string()}")

        df.to_csv("raw_acled_region.csv",  index=False)
        df_hormuz.to_csv("raw_acled_hormuz.csv", index=False)
        print("\nGuardados: raw_acled_region.csv | raw_acled_hormuz.csv")
    else:
        print("Sin datos.")
