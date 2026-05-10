import os
from dotenv import load_dotenv

load_dotenv()

# ─── CREDENCIALES ─────────────────────────────────────────────────────────────
ACLED_EMAIL    = os.getenv("ACLED_EMAIL")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD")
FIRMS_MAP_KEY  = os.getenv("FIRMS_MAP_KEY")
OPENSKY_CLIENT_ID     = os.getenv("OPENSKY_CLIENT_ID")
OPENSKY_CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET")
AISSTREAM_API_KEY     = os.getenv("AISSTREAM_API_KEY")

# ─── BOUNDING BOXES ───────────────────────────────────────────────────────────
HORMUZ_BBOX = {
    "lat_min": 25.0, "lat_max": 27.5,
    "lon_min": 55.5, "lon_max": 58.0
}
MAR_ROJO_BBOX = {
    "lat_min": 11.0, "lat_max": 20.0,
    "lon_min": 41.0, "lon_max": 45.0
}
GOLFO_BBOX = {
    "lat_min": 23.0, "lat_max": 26.0,
    "lon_min": 50.0, "lon_max": 56.0
}

# ─── PAÍSES DE INTERÉS ────────────────────────────────────────────────────────
PAISES_ACLED = "Iran|United States|Yemen|Iraq|United Arab Emirates|Oman|Israel"
PAISES_COD   = ["IRN", "US", "YEM", "IRQ", "SAU", "UAE", "OMN", "ISR", "JOR"]

# ─── KEYWORDS GDELT ───────────────────────────────────────────────────────────
KEYWORDS_GDELT = [
    "Hormuz", "Iran", "Persian Gulf", "Yemen",
    "Red Sea", "United States", "Iraq", "IRGC", "Houthi"
]

# ─── RUTAS ────────────────────────────────────────────────────────────────────
import pathlib
ROOT     = pathlib.Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
MODELS   = ROOT / "models" / "saved"
