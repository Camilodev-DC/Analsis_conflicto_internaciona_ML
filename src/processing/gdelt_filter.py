import pandas as pd

# ─── CARGAR CSV EXISTENTE ─────────────────────────────────────────────────────
print("Cargando raw_gdelt_2024_hormuz.csv...")
df = pd.read_csv("raw_gdelt_2024_hormuz.csv", low_memory=False)
print(f"Total filas cargadas: {len(df)}")

# ─── FILTROS EEUU-IRAN + HORMUZ ───────────────────────────────────────────────
PAISES_INTERES = ["IRN", "US", "YEM", "IRQ", "SYR", "JOR", "ISR", "SAU"]
KEYWORDS_GEO   = ["Iran", "Hormuz", "Persian Gulf", "Yemen", "Red Sea",
                   "United States", "Iraq", "Syria", "Jordan", "Gulf"]

# filtro por país
mask_pais = pd.Series([False] * len(df), index=df.index)
for col in ["Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode"]:
    if col in df.columns:
        mask_pais |= df[col].isin(PAISES_INTERES)

# filtro por geografía
mask_geo = pd.Series([False] * len(df), index=df.index)
for col in ["Actor1Geo_FullName", "Actor2Geo_FullName", "ActionGeo_FullName"]:
    if col in df.columns:
        for kw in KEYWORDS_GEO:
            mask_geo |= df[col].astype(str).str.contains(kw, case=False, na=False)

# filtro por actores (nombre)
mask_actor = pd.Series([False] * len(df), index=df.index)
for col in ["Actor1Name", "Actor2Name"]:
    if col in df.columns:
        for kw in ["IRAN", "UNITED STATES", "HOUTHI", "IRGC", "PENTAGON", "BIDEN", "TRUMP"]:
            mask_actor |= df[col].astype(str).str.contains(kw, case=False, na=False)

df_fil = df[mask_pais | mask_geo | mask_actor].copy()
df_fil = df_fil.drop_duplicates(subset=["GlobalEventID"])

print(f"Eventos EEUU-Iran+Hormuz: {len(df_fil)}")

# ─── LIMPIAR Y ENRIQUECER ─────────────────────────────────────────────────────
df_fil["Day"] = df_fil["Day"].astype(str)
df_fil["fecha"] = pd.to_datetime(df_fil["Day"], format="%Y%m%d", errors="coerce")
df_fil["GoldsteinScale"] = pd.to_numeric(df_fil["GoldsteinScale"], errors="coerce")
df_fil["AvgTone"]        = pd.to_numeric(df_fil["AvgTone"],        errors="coerce")
df_fil["NumMentions"]    = pd.to_numeric(df_fil["NumMentions"],    errors="coerce")
df_fil["NumArticles"]    = pd.to_numeric(df_fil["NumArticles"],    errors="coerce")
df_fil["ActionGeo_Lat"]  = pd.to_numeric(df_fil["ActionGeo_Lat"], errors="coerce")
df_fil["ActionGeo_Long"] = pd.to_numeric(df_fil["ActionGeo_Long"],errors="coerce")

# clasificar región
def clasificar_region(row):
    lat = row["ActionGeo_Lat"]
    lon = row["ActionGeo_Long"]
    if pd.isna(lat) or pd.isna(lon):
        return "desconocida"
    if 25.0 <= lat <= 27.5 and 55.5 <= lon <= 58.0:
        return "hormuz"
    if 11.0 <= lat <= 20.0 and 41.0 <= lon <= 45.0:
        return "mar_rojo"
    if 23.0 <= lat <= 26.0 and 50.0 <= lon <= 56.0:
        return "golfo_persico"
    return "region_amplia"

df_fil["region"] = df_fil.apply(clasificar_region, axis=1)

# ─── AGREGADO DIARIO (para el modelo ML) ─────────────────────────────────────
df_dia = df_fil.groupby("fecha").agg(
    gdelt_n_eventos      = ("GlobalEventID", "count"),
    gdelt_tono_promedio  = ("GoldsteinScale", "mean"),
    gdelt_tono_min       = ("GoldsteinScale", "min"),
    gdelt_avgtone        = ("AvgTone", "mean"),
    gdelt_n_menciones    = ("NumMentions", "sum"),
    gdelt_n_articulos    = ("NumArticles", "sum"),
).reset_index()

# ─── RESUMEN ─────────────────────────────────────────────────────────────────
print(f"\nPeriodo: {df_fil['fecha'].min().date()} a {df_fil['fecha'].max().date()}")
print(f"Regiones:\n{df_fil['region'].value_counts()}")
print(f"\nTono promedio GoldsteinScale: {df_fil['GoldsteinScale'].mean():.2f}")
print(f"Tono promedio AvgTone: {df_fil['AvgTone'].mean():.2f}")
print(f"\nTop 5 dias con mas eventos:")
print(df_dia.nlargest(5, "gdelt_n_eventos")[["fecha","gdelt_n_eventos","gdelt_tono_promedio"]].to_string())

# ─── GUARDAR ─────────────────────────────────────────────────────────────────
df_fil.to_csv("gdelt_eeuu_iran_filtrado.csv", index=False)
df_dia.to_csv("gdelt_agregado_diario.csv", index=False)
print(f"\nGuardados:")
print(f"  gdelt_eeuu_iran_filtrado.csv  ({len(df_fil)} eventos)")
print(f"  gdelt_agregado_diario.csv     ({len(df_dia)} dias)")
