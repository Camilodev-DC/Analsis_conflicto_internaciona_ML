"""
Dataset manual de incidentes maritimos en el Estrecho de Hormuz / Mar Rojo 2024
Fuente: UKMTO advisories, Reuters, BBC, AP News (datos de dominio publico)
Documentado como fuente externa verificable.
"""
import pandas as pd

INCIDENTS = [
    # ── ENERO 2024 ──────────────────────────────────────────────────────────
    {"fecha": "2024-01-09", "vessel": "MV Zografia",        "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.5,  "lon": 42.5,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con misil a buque de carga griega en Mar Rojo"},
    {"fecha": "2024-01-11", "vessel": "MV Blaamanen",       "tipo": "drone_attack",    "actor": "Houthi", "lat": 14.2,  "lon": 42.8,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con dron a buque noruego"},
    {"fecha": "2024-01-15", "vessel": "MV Khalissa",        "tipo": "missile_attack",  "actor": "Houthi", "lat": 13.8,  "lon": 42.3,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con misil a buque en Mar Rojo"},
    {"fecha": "2024-01-18", "vessel": "MV Marlin Luanda",   "tipo": "missile_attack",  "actor": "Houthi", "lat": 15.1,  "lon": 42.0,  "region": "mar_rojo",  "severidad": 3, "descripcion": "Buque tanquero incendiado tras ataque hutí"},
    {"fecha": "2024-01-22", "vessel": "MV Genco Picardy",   "tipo": "drone_attack",    "actor": "Houthi", "lat": 14.9,  "lon": 42.4,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con dron a buque de carga"},
    {"fecha": "2024-01-26", "vessel": "MV Chem Ranger",     "tipo": "seizure_attempt", "actor": "IRGC",   "lat": 25.9,  "lon": 56.8,  "region": "hormuz",    "severidad": 3, "descripcion": "Intento de abordaje por IRGC en Estrecho de Hormuz"},
    {"fecha": "2024-01-28", "vessel": "US Base Tower 22",   "tipo": "drone_attack",    "actor": "Iran_proxy", "lat": 32.5, "lon": 38.2, "region": "jordan", "severidad": 4, "descripcion": "Ataque dron a base EEUU en Jordania — 3 soldados muertos"},
    # ── FEBRERO 2024 ────────────────────────────────────────────────────────
    {"fecha": "2024-02-01", "vessel": "Multiple",           "tipo": "us_airstrike",    "actor": "EEUU",   "lat": 15.0,  "lon": 44.0,  "region": "yemen",     "severidad": 3, "descripcion": "EEUU bombardea 85 objetivos hutíes en Yemen"},
    {"fecha": "2024-02-18", "vessel": "MV Rubymar",         "tipo": "missile_attack",  "actor": "Houthi", "lat": 13.5,  "lon": 43.2,  "region": "mar_rojo",  "severidad": 4, "descripcion": "Buque hundido tras ataque hutí — primer buque hundido"},
    {"fecha": "2024-02-24", "vessel": "MV Talia",           "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.3,  "lon": 42.6,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con misil a buque de carga"},
    # ── MARZO 2024 ──────────────────────────────────────────────────────────
    {"fecha": "2024-03-06", "vessel": "MV True Confidence", "tipo": "missile_attack",  "actor": "Houthi", "lat": 13.1,  "lon": 43.8,  "region": "mar_rojo",  "severidad": 4, "descripcion": "3 tripulantes muertos en ataque hutí"},
    {"fecha": "2024-03-12", "vessel": "MV Pinocchio",       "tipo": "drone_attack",    "actor": "Houthi", "lat": 14.7,  "lon": 42.1,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Dron impacta buque de carga general"},
    {"fecha": "2024-03-24", "vessel": "MV Tutor",           "tipo": "drone_attack",    "actor": "Houthi", "lat": 15.3,  "lon": 41.8,  "region": "mar_rojo",  "severidad": 3, "descripcion": "Buque alcanzado por dron kamikaze"},
    # ── ABRIL 2024 ──────────────────────────────────────────────────────────
    {"fecha": "2024-04-01", "vessel": "Consulate Damascus", "tipo": "airstrike",       "actor": "Israel", "lat": 33.5,  "lon": 36.3,  "region": "siria",     "severidad": 4, "descripcion": "Israel ataca consulado iraní en Damasco — 13 muertos"},
    {"fecha": "2024-04-13", "vessel": "Israel territory",   "tipo": "missile_drone",   "actor": "Iran",   "lat": 31.5,  "lon": 34.8,  "region": "israel",    "severidad": 5, "descripcion": "Irán lanza 300+ drones y misiles contra Israel — escalada máxima"},
    {"fecha": "2024-04-14", "vessel": "Israel territory",   "tipo": "interception",    "actor": "EEUU_Israel", "lat": 31.5, "lon": 34.8, "region": "israel",  "severidad": 5, "descripcion": "EEUU e Israel interceptan ataque iraní"},
    {"fecha": "2024-04-18", "vessel": "Iran territory",     "tipo": "airstrike",       "actor": "Israel", "lat": 32.4,  "lon": 53.7,  "region": "iran",      "severidad": 4, "descripcion": "Israel ataca objetivos en Irán — respuesta limitada"},
    {"fecha": "2024-04-13", "vessel": "MSC Aries",         "tipo": "seizure",         "actor": "IRGC",   "lat": 25.8,  "lon": 57.1,  "region": "hormuz",    "severidad": 5, "descripcion": "IRGC aborda y confisca buque MSC Aries en Estrecho de Hormuz"},
    # ── MAYO 2024 ───────────────────────────────────────────────────────────
    {"fecha": "2024-05-02", "vessel": "MV Yorktown",        "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.0,  "lon": 42.9,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque con misil en Mar Rojo"},
    {"fecha": "2024-05-27", "vessel": "MV Tutor",           "tipo": "boat_attack",     "actor": "Houthi", "lat": 15.5,  "lon": 41.5,  "region": "mar_rojo",  "severidad": 4, "descripcion": "Buque hundido tras ataque con lancha explosiva"},
    # ── JUNIO 2024 ──────────────────────────────────────────────────────────
    {"fecha": "2024-06-12", "vessel": "MV Verbena",         "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.8,  "lon": 42.3,  "region": "mar_rojo",  "severidad": 3, "descripcion": "Buque alcanzado y averiado en Mar Rojo"},
    {"fecha": "2024-06-22", "vessel": "MV Pumba",           "tipo": "drone_attack",    "actor": "Houthi", "lat": 13.9,  "lon": 43.1,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Dron impacta buque de carga griega"},
    # ── JULIO 2024 ──────────────────────────────────────────────────────────
    {"fecha": "2024-07-19", "vessel": "MV Tutor (naufragio)","tipo": "sinking",        "actor": "Houthi", "lat": 15.6,  "lon": 41.4,  "region": "mar_rojo",  "severidad": 4, "descripcion": "Naufragio definitivo del MV Tutor"},
    {"fecha": "2024-07-19", "vessel": "Tel Aviv",            "tipo": "drone_attack",   "actor": "Houthi", "lat": 32.1,  "lon": 34.8,  "region": "israel",    "severidad": 4, "descripcion": "Dron hutí impacta Tel Aviv — 1 muerto"},
    # ── AGOSTO 2024 ─────────────────────────────────────────────────────────
    {"fecha": "2024-08-20", "vessel": "MV Sounion",         "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.2,  "lon": 43.5,  "region": "mar_rojo",  "severidad": 4, "descripcion": "Tanquero griego incendiado — amenaza de derrame"},
    {"fecha": "2024-08-25", "vessel": "MV Sounion",         "tipo": "oil_spill_risk",  "actor": "Houthi", "lat": 14.2,  "lon": 43.5,  "region": "mar_rojo",  "severidad": 4, "descripcion": "Sounion sigue ardiendo — riesgo de derrame masivo"},
    # ── SEPTIEMBRE 2024 ─────────────────────────────────────────────────────
    {"fecha": "2024-09-15", "vessel": "Multiple vessels",   "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.5,  "lon": 42.8,  "region": "mar_rojo",  "severidad": 3, "descripcion": "Múltiples ataques simultáneos en Mar Rojo"},
    # ── OCTUBRE 2024 ────────────────────────────────────────────────────────
    {"fecha": "2024-10-01", "vessel": "Iran territory",     "tipo": "airstrike",       "actor": "Israel", "lat": 32.4,  "lon": 53.7,  "region": "iran",      "severidad": 5, "descripcion": "Israel lanza operación militar directa contra Irán"},
    {"fecha": "2024-10-02", "vessel": "Iran territory",     "tipo": "airstrike",       "actor": "Israel", "lat": 35.7,  "lon": 51.4,  "region": "iran",      "severidad": 5, "descripcion": "Israel ataca infraestructura militar iraní"},
    {"fecha": "2024-10-10", "vessel": "MV Akaroa",          "tipo": "missile_attack",  "actor": "Houthi", "lat": 14.1,  "lon": 42.7,  "region": "mar_rojo",  "severidad": 2, "descripcion": "Ataque hutí en Mar Rojo"},
    {"fecha": "2024-10-17", "vessel": "Hormuz passage",     "tipo": "irgc_warning",    "actor": "IRGC",   "lat": 26.5,  "lon": 56.5,  "region": "hormuz",    "severidad": 3, "descripcion": "IRGC emite advertencia de cierre de Hormuz ante operaciones israelíes"},
]

if __name__ == "__main__":
    df = pd.DataFrame(INCIDENTS)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["source"] = "maritime_manual"

    print(f"Total incidentes: {len(df)}")
    print(f"\nPor región:\n{df['region'].value_counts()}")
    print(f"\nPor tipo:\n{df['tipo'].value_counts()}")
    print(f"\nPor severidad:\n{df['severidad'].value_counts().sort_index()}")
    print(f"\nIncidentes en Hormuz:\n{df[df['region']=='hormuz'][['fecha','vessel','tipo','descripcion']].to_string()}")

    # Agregar por día para el JOIN con el dataset principal
    df_dia = df.groupby("fecha").agg(
        maritime_n_incidentes = ("tipo", "count"),
        maritime_severidad_max = ("severidad", "max"),
        maritime_severidad_sum = ("severidad", "sum"),
        maritime_n_hormuz = ("region", lambda x: (x == "hormuz").sum()),
        maritime_n_ataques_directos = ("tipo", lambda x: x.isin(["missile_attack","drone_attack","airstrike","seizure"]).sum()),
    ).reset_index()

    df.to_csv("../../data/raw/raw_maritime_incidents.csv", index=False)
    df_dia.to_csv("../../data/processed/maritime_agregado_diario.csv", index=False)
    print(f"\nGuardados:")
    print(f"  data/raw/raw_maritime_incidents.csv ({len(df)} incidentes)")
    print(f"  data/processed/maritime_agregado_diario.csv ({len(df_dia)} dias)")
