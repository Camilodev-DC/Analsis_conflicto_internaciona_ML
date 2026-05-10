"""
EDA Completo — Sistema de Inteligencia Multifuente Estrecho de Hormuz 2024
==========================================================================
Secciones:
  0. Setup y carga de datos
  1. EDA ACLED (región + Hormuz)
  2. EDA GDELT
  3. EDA OpenSky
  4. EDA Maritime Incidents (manual)
  5. EDA UKMTO 2024
  6. EDA Dataset Integrado
  7. Análisis del Target (nivel_riesgo)
  8. Correlaciones y multicolinealidad
  9. Series de tiempo multifuente
 10. Resumen de hallazgos
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
RAW   = ROOT / "data" / "raw"
PROC  = ROOT / "data" / "processed"
FIGS  = ROOT / "docs" / "eda_figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Estilo global ─────────────────────────────────────────────────────────────
PALETTE = {"BAJO": "#2ecc71", "MEDIO": "#f39c12", "ALTO": "#e74c3c"}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})

def save(fig, name):
    p = FIGS / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [guardado] {p.name}")

# =============================================================================
# 0. CARGA
# =============================================================================
print("\n" + "="*70)
print("0. CARGA DE DATOS")
print("="*70)

df_int   = pd.read_csv(PROC / "dataset_integrado.csv", parse_dates=["fecha"])
df_acled = pd.read_csv(RAW  / "raw_acled_region.csv",  parse_dates=["event_date"])
df_horm  = pd.read_csv(RAW  / "raw_acled_hormuz.csv",  parse_dates=["event_date"])
df_gdelt = pd.read_csv(RAW  / "raw_gdelt_2024_hormuz.csv", low_memory=False,
                        encoding="utf-8", encoding_errors="replace")
df_sky   = pd.read_csv(RAW  / "raw_opensky_historico.csv")
df_mar   = pd.read_csv(RAW  / "raw_maritime_incidents.csv", parse_dates=["fecha"])
df_ukmto = pd.read_csv(RAW  / "raw_ukmto_2024_archivo.csv", parse_dates=["fecha"])

# Limpieza mínima
df_gdelt["GoldsteinScale"] = pd.to_numeric(df_gdelt["GoldsteinScale"], errors="coerce")
df_gdelt["NumMentions"]    = pd.to_numeric(df_gdelt["NumMentions"],    errors="coerce")
df_gdelt["fecha"] = pd.to_datetime(df_gdelt["Day"].astype(str), format="%Y%m%d", errors="coerce")
df_sky["on_ground"] = df_sky["on_ground"].astype(str).str.lower() == "true"
df_sky["baro_altitude"] = pd.to_numeric(df_sky["baro_altitude"], errors="coerce")
df_sky["velocity"]      = pd.to_numeric(df_sky["velocity"],      errors="coerce")
df_ukmto["tipo"] = df_ukmto["nombre"].apply(
    lambda n: "Attack" if "ATTACK" in str(n).upper()
    else ("Hijack" if "HIJACK" in str(n).upper()
    else ("Suspicious" if "SUSPICIOUS" in str(n).upper() else "Warning"))
)

print(f"  Dataset integrado : {df_int.shape}")
print(f"  ACLED región      : {df_acled.shape}")
print(f"  ACLED Hormuz      : {df_horm.shape}")
print(f"  GDELT             : {df_gdelt.shape}")
print(f"  OpenSky           : {df_sky.shape}")
print(f"  Maritime manual   : {df_mar.shape}")
print(f"  UKMTO 2024        : {df_ukmto.shape}")

# =============================================================================
# 1. EDA ACLED — REGIÓN
# =============================================================================
print("\n" + "="*70)
print("1. EDA ACLED REGIÓN")
print("="*70)

# 1a. Distribución de tipos de evento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ACLED Región — Distribución de Eventos (37 722 eventos, 2024)", fontsize=13, fontweight="bold")

vc = df_acled["event_type"].value_counts()
colors_et = sns.color_palette("Set2", len(vc))
axes[0].barh(vc.index, vc.values, color=colors_et)
axes[0].set_xlabel("N° Eventos")
axes[0].set_title("Tipo de Evento")
for i, v in enumerate(vc.values):
    axes[0].text(v + 100, i, f"{v:,}", va="center", fontsize=9)

vc_c = df_acled["country"].value_counts().head(8)
axes[1].barh(vc_c.index, vc_c.values, color=sns.color_palette("Set3", len(vc_c)))
axes[1].set_xlabel("N° Eventos")
axes[1].set_title("Top 8 Países")
for i, v in enumerate(vc_c.values):
    axes[1].text(v + 50, i, f"{v:,}", va="center", fontsize=9)

plt.tight_layout()
save(fig, "01_acled_distribucion")

# 1b. Serie temporal diaria por tipo
fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle("ACLED Región — Eventos Violentos por Día (2024)", fontsize=13, fontweight="bold")

violent = ["Explosions/Remote violence", "Battles", "Violence against civilians"]
df_viol = df_acled[df_acled["event_type"].isin(violent)].copy()
df_viol["date"] = df_viol["event_date"].dt.date
daily_viol = df_viol.groupby(["date", "event_type"]).size().unstack(fill_value=0)
daily_viol.index = pd.to_datetime(daily_viol.index)
daily_viol.plot(ax=ax, colormap="Set1", linewidth=1.2, alpha=0.85)

# Marcar eventos clave
eventos_clave = {
    "2024-01-28": "Torre 22",
    "2024-04-13": "Iran→Israel",
    "2024-10-01": "Israel→Iran",
}
for fecha, label in eventos_clave.items():
    ax.axvline(pd.to_datetime(fecha), color="black", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.text(pd.to_datetime(fecha), ax.get_ylim()[1]*0.85, label,
            rotation=90, fontsize=8, ha="right", color="black")

ax.set_xlabel("")
ax.set_ylabel("N° Eventos")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend(title="Tipo", loc="upper left", fontsize=8)
plt.tight_layout()
save(fig, "02_acled_serie_temporal")

# 1c. ACLED Hormuz — actores y tipos
print(f"\n  ACLED Hormuz: {len(df_horm)} eventos")
print(f"  Tipos:\n{df_horm['event_type'].value_counts().to_string()}")
print(f"  Actores principales:\n{df_horm['actor1'].value_counts().head(8).to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ACLED Hormuz — 81 eventos directos en el Estrecho (2024)", fontsize=13, fontweight="bold")

df_horm["event_type"].value_counts().plot.bar(ax=axes[0], color=sns.color_palette("Reds_r", 6), rot=30)
axes[0].set_title("Tipo de Evento")
axes[0].set_ylabel("N°")

df_horm["actor1"].value_counts().head(8).plot.barh(ax=axes[1], color=sns.color_palette("Blues_r", 8))
axes[1].set_title("Actor Principal (top 8)")
axes[1].set_xlabel("N°")

df_horm["month"] = df_horm["event_date"].dt.month
monthly = df_horm.groupby("month")["event_type"].count()
monthly.index = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"][:len(monthly)]
monthly.plot.bar(ax=axes[2], color="#e74c3c", rot=45)
axes[2].set_title("Eventos por Mes")
axes[2].set_ylabel("N°")

plt.tight_layout()
save(fig, "03_acled_hormuz_detalle")

# =============================================================================
# 2. EDA GDELT
# =============================================================================
print("\n" + "="*70)
print("2. EDA GDELT")
print("="*70)

print(f"  Goldstein: mean={df_gdelt['GoldsteinScale'].mean():.2f} | "
      f"std={df_gdelt['GoldsteinScale'].std():.2f} | "
      f"min={df_gdelt['GoldsteinScale'].min():.1f} | max={df_gdelt['GoldsteinScale'].max():.1f}")
print(f"  QuadClass dist:\n{df_gdelt['QuadClass'].value_counts().sort_index().to_string()}")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("GDELT — Cobertura Mediática del Conflicto (28 687 eventos, Abr–Oct 2024)",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig)

# Goldstein distribution
ax1 = fig.add_subplot(gs[0, 0])
df_gdelt["GoldsteinScale"].dropna().hist(ax=ax1, bins=40, color="#3498db", edgecolor="white")
ax1.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Neutro")
ax1.axvline(df_gdelt["GoldsteinScale"].mean(), color="orange", linestyle="--", linewidth=1.2,
            label=f"Media={df_gdelt['GoldsteinScale'].mean():.1f}")
ax1.set_title("Distribución Goldstein Scale")
ax1.set_xlabel("Goldstein (−10 conflicto → +10 cooperación)")
ax1.legend(fontsize=8)

# QuadClass
ax2 = fig.add_subplot(gs[0, 1])
qc_labels = {1: "Verbal\nCoop", 2: "Mat.\nCoop", 3: "Verbal\nConf", 4: "Mat.\nConf"}
qc = df_gdelt["QuadClass"].value_counts().sort_index()
bars = ax2.bar([qc_labels.get(i, str(i)) for i in qc.index], qc.values,
               color=["#2ecc71","#3498db","#f39c12","#e74c3c"])
ax2.set_title("Distribución QuadClass")
ax2.set_ylabel("N° Eventos")
for bar, val in zip(bars, qc.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", fontsize=9)

# NumMentions distribution (log)
ax3 = fig.add_subplot(gs[0, 2])
mentions = df_gdelt["NumMentions"].dropna()
ax3.hist(np.log1p(mentions), bins=40, color="#9b59b6", edgecolor="white")
ax3.set_title("Distribución NumMentions (log1p)")
ax3.set_xlabel("log(1 + NumMentions)")

# Serie temporal Goldstein
ax4 = fig.add_subplot(gs[1, :])
daily_gdelt = df_gdelt.dropna(subset=["fecha"]).groupby("fecha").agg(
    goldstein_mean=("GoldsteinScale", "mean"),
    n_conflicto=("QuadClass", lambda x: (x == 3).sum()),
    n_menciones=("NumMentions", "sum")
).reset_index()
daily_gdelt = daily_gdelt[daily_gdelt["fecha"].dt.year == 2024]

ax4b = ax4.twinx()
ax4.plot(daily_gdelt["fecha"], daily_gdelt["goldstein_mean"],
         color="#3498db", linewidth=1.5, label="Goldstein medio", alpha=0.9)
ax4.fill_between(daily_gdelt["fecha"], daily_gdelt["goldstein_mean"], 0,
                  where=daily_gdelt["goldstein_mean"] < 0,
                  alpha=0.2, color="red", label="Zona conflicto")
ax4b.bar(daily_gdelt["fecha"], daily_gdelt["n_conflicto"],
         color="#e74c3c", alpha=0.4, label="N° eventos conflicto")
ax4.set_title("Tono Mediático Diario (Goldstein) y Eventos de Conflicto")
ax4.set_ylabel("Goldstein Scale", color="#3498db")
ax4b.set_ylabel("N° Eventos Conflicto", color="#e74c3c")
ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

plt.tight_layout()
save(fig, "04_gdelt_eda")

# =============================================================================
# 3. EDA OPENSKY
# =============================================================================
print("\n" + "="*70)
print("3. EDA OPENSKY")
print("="*70)

print(f"  Vuelos totales: {len(df_sky)}")
print(f"  Países de origen:\n{df_sky['origin_country'].value_counts().head(10).to_string()}")
print(f"  Altitud media: {df_sky['baro_altitude'].mean():.0f}m")
print(f"  Velocidad media: {df_sky['velocity'].mean():.1f} m/s")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("OpenSky — Tráfico Aéreo sobre Hormuz/Golfo (9 fechas clave, 59 vuelos)",
             fontsize=13, fontweight="bold")

# Vuelos por fecha
vc_fecha = df_sky.groupby("fecha").size().reset_index(name="n")
label_map = {
    "2024-04-13": "Iran→Israel\n(D1)", "2024-04-14": "Iran→Israel\n(D2)",
    "2024-04-15": "Post-ataque", "2024-01-28": "Torre 22",
    "2024-10-01": "Israel→Iran", "2024-10-02": "Post-resp.",
    "2024-03-01": "Control\nMar", "2024-06-15": "Control\nJun",
    "2024-08-10": "Control\nAgo"
}
colors_sky = ["#e74c3c" if "ataque" in label_map.get(f,"").lower() or "Israel" in label_map.get(f,"")
              else "#2ecc71" if "Control" in label_map.get(f,"")
              else "#f39c12"
              for f in vc_fecha["fecha"]]
bars = axes[0,0].bar([label_map.get(f, f) for f in vc_fecha["fecha"]], vc_fecha["n"],
                      color=["#e74c3c","#e74c3c","#f39c12","#f39c12","#e74c3c","#f39c12",
                             "#2ecc71","#2ecc71","#2ecc71"])
axes[0,0].set_title("Vuelos detectados por fecha")
axes[0,0].set_ylabel("N° Vuelos")
for bar, val in zip(bars, vc_fecha["n"]):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(val), ha="center", fontsize=9)

# País de origen
vc_pais = df_sky["origin_country"].value_counts().head(8)
axes[0,1].barh(vc_pais.index, vc_pais.values, color=sns.color_palette("Set2", len(vc_pais)))
axes[0,1].set_title("País de origen (top 8)")
axes[0,1].set_xlabel("N° Vuelos")

# Distribución altitud
df_sky_air = df_sky[~df_sky["on_ground"]]
axes[1,0].hist(df_sky_air["baro_altitude"].dropna(), bins=20, color="#3498db", edgecolor="white")
axes[1,0].set_title("Distribución de Altitud Barométrica (vuelos en aire)")
axes[1,0].set_xlabel("Altitud (metros)")
axes[1,0].set_ylabel("N° Vuelos")

# Velocidad vs altitud
sc = axes[1,1].scatter(df_sky["velocity"].dropna(), df_sky["baro_altitude"].dropna(),
                        c=df_sky["on_ground"].map({True:"#e74c3c", False:"#2ecc71"}),
                        alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
axes[1,1].set_title("Velocidad vs Altitud")
axes[1,1].set_xlabel("Velocidad (m/s)")
axes[1,1].set_ylabel("Altitud (m)")
from matplotlib.patches import Patch
axes[1,1].legend(handles=[Patch(color="#2ecc71", label="Airborne"),
                            Patch(color="#e74c3c", label="On Ground")], fontsize=9)

plt.tight_layout()
save(fig, "05_opensky_eda")

# =============================================================================
# 4. EDA MARITIME INCIDENTS
# =============================================================================
print("\n" + "="*70)
print("4. EDA MARITIME INCIDENTS (manual)")
print("="*70)

print(f"  Total incidentes: {len(df_mar)}")
print(f"  Tipos:\n{df_mar['tipo'].value_counts().to_string()}")
print(f"  Actores:\n{df_mar['actor'].value_counts().to_string()}")
print(f"  Regiones:\n{df_mar['region'].value_counts().to_string()}")
print(f"  Severidad media: {df_mar['severidad'].mean():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Maritime Incidents Manual — 31 Incidentes de Alta Relevancia (2024)",
             fontsize=13, fontweight="bold")

# Tipo
vc_tipo = df_mar["tipo"].value_counts()
axes[0,0].bar(vc_tipo.index, vc_tipo.values, color=sns.color_palette("Reds_r", len(vc_tipo)))
axes[0,0].set_title("Tipo de Incidente")
axes[0,0].set_ylabel("N°")
axes[0,0].tick_params(axis="x", rotation=40)

# Actor
vc_actor = df_mar["actor"].value_counts()
axes[0,1].pie(vc_actor.values, labels=vc_actor.index, autopct="%1.0f%%",
              colors=sns.color_palette("Set2", len(vc_actor)), startangle=140)
axes[0,1].set_title("Actor Responsable")

# Severidad por región
sev_region = df_mar.groupby("region")["severidad"].mean().sort_values(ascending=False)
axes[1,0].bar(sev_region.index, sev_region.values,
              color=[PALETTE["ALTO"] if v >= 4 else PALETTE["MEDIO"] if v >= 3 else PALETTE["BAJO"]
                     for v in sev_region.values])
axes[1,0].set_title("Severidad Media por Región")
axes[1,0].set_ylabel("Severidad (1–5)")
axes[1,0].axhline(3, color="orange", linestyle="--", linewidth=1, label="Umbral ALTO")
axes[1,0].legend(fontsize=8)
axes[1,0].tick_params(axis="x", rotation=30)

# Serie temporal de severidad
df_mar_sorted = df_mar.sort_values("fecha")
axes[1,1].scatter(df_mar_sorted["fecha"], df_mar_sorted["severidad"],
                  c=[PALETTE["ALTO"] if s >= 4 else PALETTE["MEDIO"] if s >= 2 else PALETTE["BAJO"]
                     for s in df_mar_sorted["severidad"]],
                  s=80, zorder=5, edgecolors="white", linewidth=0.5)
axes[1,1].plot(df_mar_sorted["fecha"], df_mar_sorted["severidad"],
               color="gray", linewidth=0.8, alpha=0.5)
axes[1,1].set_title("Severidad de Incidentes a lo largo de 2024")
axes[1,1].set_ylabel("Severidad")
axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
from matplotlib.patches import Patch
axes[1,1].legend(handles=[Patch(color=PALETTE[k], label=k) for k in ["BAJO","MEDIO","ALTO"]], fontsize=8)

plt.tight_layout()
save(fig, "06_maritime_eda")

# =============================================================================
# 5. EDA UKMTO 2024
# =============================================================================
print("\n" + "="*70)
print("5. EDA UKMTO 2024")
print("="*70)

df_ukmto_orig = df_ukmto[~df_ukmto["es_update"]].copy()
print(f"  Total registros: {len(df_ukmto)}")
print(f"  Incidentes originales: {len(df_ukmto_orig)}")
print(f"  Tipos:\n{df_ukmto_orig['tipo'].value_counts().to_string()}")

MONTHS_ORDER = ["January","February","March","April","May","June",
                "July","August","September","October","November"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("UKMTO 2024 — Archivo Histórico Oficial (246 reportes, 99 incidentes)",
             fontsize=13, fontweight="bold")

# Incidentes por mes
monthly_ukmto = df_ukmto_orig.groupby("mes").size().reindex(MONTHS_ORDER).fillna(0)
bars = axes[0].bar(range(len(monthly_ukmto)), monthly_ukmto.values,
                   color=sns.color_palette("YlOrRd", len(monthly_ukmto)))
axes[0].set_xticks(range(len(monthly_ukmto)))
axes[0].set_xticklabels([m[:3] for m in MONTHS_ORDER], rotation=45)
axes[0].set_title("Incidentes Originales por Mes")
axes[0].set_ylabel("N°")
for bar, val in zip(bars, monthly_ukmto.values):
    if val > 0:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     int(val), ha="center", fontsize=9)

# Tipo de incidente
vc_tipo_ukmto = df_ukmto_orig["tipo"].value_counts()
colors_u = [PALETTE["ALTO"] if t == "Attack" else
            PALETTE["BAJO"] if t == "Warning" else
            PALETTE["MEDIO"] for t in vc_tipo_ukmto.index]
axes[1].bar(vc_tipo_ukmto.index, vc_tipo_ukmto.values, color=colors_u)
axes[1].set_title("Tipo de Incidente")
axes[1].set_ylabel("N°")
axes[1].tick_params(axis="x", rotation=25)
for i, val in enumerate(vc_tipo_ukmto.values):
    axes[1].text(i, val + 0.3, str(val), ha="center", fontsize=10, fontweight="bold")

# Attacks vs Warnings por mes
pivot_ukmto = df_ukmto_orig.pivot_table(index="mes", columns="tipo", values="referencia",
                                         aggfunc="count", fill_value=0).reindex(MONTHS_ORDER)
if "Attack" in pivot_ukmto.columns:
    axes[2].bar(range(len(pivot_ukmto)), pivot_ukmto.get("Attack", 0),
                label="Attack", color=PALETTE["ALTO"], alpha=0.9)
if "Warning" in pivot_ukmto.columns:
    attacks = pivot_ukmto.get("Attack", pd.Series([0]*len(pivot_ukmto))).values
    axes[2].bar(range(len(pivot_ukmto)), pivot_ukmto.get("Warning", 0),
                bottom=attacks, label="Warning", color="#f39c12", alpha=0.8)
axes[2].set_xticks(range(len(pivot_ukmto)))
axes[2].set_xticklabels([m[:3] for m in MONTHS_ORDER], rotation=45)
axes[2].set_title("Attacks vs Warnings por Mes")
axes[2].set_ylabel("N°")
axes[2].legend(fontsize=8)

plt.tight_layout()
save(fig, "07_ukmto_eda")

# =============================================================================
# 6. EDA DATASET INTEGRADO
# =============================================================================
print("\n" + "="*70)
print("6. EDA DATASET INTEGRADO")
print("="*70)

print(f"  Shape: {df_int.shape}")
print(f"\n  Valores nulos por columna:")
nulls = df_int.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  Sin nulos")
print(f"\n  Estadísticas descriptivas (features clave):")
key_cols = ["acled_region_n_violentos","acled_hormuz_n_violentos","gdelt_goldstein_mean",
            "gdelt_n_conflicto","maritime_severidad_max","ukmto_n_attacks"]
print(df_int[key_cols].describe().round(2).to_string())

# 6a. Distribuciones de features clave
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Dataset Integrado — Distribuciones de Features Clave (366 días, 2024)",
             fontsize=13, fontweight="bold")

feature_info = [
    ("acled_region_n_violentos", "ACLED Violentos Región", "#e74c3c"),
    ("acled_hormuz_n_violentos", "ACLED Violentos Hormuz", "#c0392b"),
    ("gdelt_goldstein_mean",     "GDELT Goldstein Medio",  "#3498db"),
    ("gdelt_n_conflicto",        "GDELT Eventos Conflicto","#9b59b6"),
    ("ukmto_n_attacks",          "UKMTO Ataques/día",      "#e67e22"),
    ("maritime_severidad_max",   "Maritime Severidad Máx", "#2ecc71"),
]
for ax, (col, label, color) in zip(axes.flat, feature_info):
    data = df_int[col].dropna()
    ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Media={data.mean():.1f}")
    ax.axvline(data.median(), color="gray", linestyle=":", linewidth=1.2,
               label=f"Mediana={data.median():.1f}")
    ax.set_title(label)
    ax.set_ylabel("Días")
    ax.legend(fontsize=7)

plt.tight_layout()
save(fig, "08_integrado_distribuciones")

# 6b. Boxplots por nivel_riesgo
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Dataset Integrado — Features por Nivel de Riesgo",
             fontsize=13, fontweight="bold")

order = ["BAJO", "MEDIO", "ALTO"]
for ax, (col, label, _) in zip(axes.flat, feature_info):
    data_by_class = [df_int[df_int["nivel_riesgo"] == c][col].dropna() for c in order]
    bp = ax.boxplot(data_by_class, labels=order, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], [PALETTE[c] for c in order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label)
    ax.set_ylabel("Valor")

plt.tight_layout()
save(fig, "09_boxplots_por_riesgo")

# =============================================================================
# 7. ANÁLISIS DEL TARGET
# =============================================================================
print("\n" + "="*70)
print("7. ANÁLISIS DEL TARGET — nivel_riesgo")
print("="*70)

dist = df_int["nivel_riesgo"].value_counts()
print(f"  Distribución:\n{dist.to_string()}")
print(f"  Imbalance ratio (BAJO/ALTO): {dist['BAJO']/dist['ALTO']:.1f}x")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Análisis del Target — nivel_riesgo (BAJO / MEDIO / ALTO)",
             fontsize=13, fontweight="bold")

# Distribución general
bars = axes[0].bar(order, [dist.get(c, 0) for c in order],
                   color=[PALETTE[c] for c in order], edgecolor="white", linewidth=1.5)
axes[0].set_title("Distribución General")
axes[0].set_ylabel("N° Días")
for bar, val in zip(bars, [dist.get(c, 0) for c in order]):
    pct = val / len(df_int) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")

# Distribución mensual del target
monthly_target = df_int.groupby([df_int["fecha"].dt.month, "nivel_riesgo"]).size().unstack(fill_value=0)
monthly_target.index.name = "mes"
month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
bottom = np.zeros(len(monthly_target))
for clase in order:
    if clase in monthly_target.columns:
        vals = monthly_target[clase].values
        axes[1].bar(monthly_target.index, vals, bottom=bottom,
                    label=clase, color=PALETTE[clase], alpha=0.85)
        bottom += vals
axes[1].set_xticks(monthly_target.index)
axes[1].set_xticklabels([month_names[i-1] for i in monthly_target.index], rotation=45)
axes[1].set_title("Distribución Mensual del Target")
axes[1].set_ylabel("N° Días")
axes[1].legend(fontsize=8)

# Heatmap semana × mes
df_int["dia_semana_name"] = df_int["dia_semana"].map(
    {0:"Lun",1:"Mar",2:"Mie",3:"Jue",4:"Vie",5:"Sab",6:"Dom"})
df_int["nivel_num"] = df_int["nivel_riesgo"].map({"BAJO":0,"MEDIO":1,"ALTO":2})
pivot_hw = df_int.pivot_table(index="dia_semana_name", columns="mes",
                               values="nivel_num", aggfunc="mean")
pivot_hw = pivot_hw.reindex(["Lun","Mar","Mie","Jue","Vie","Sab","Dom"])
pivot_hw.columns = [month_names[c-1] for c in pivot_hw.columns]
sns.heatmap(pivot_hw, ax=axes[2], cmap="RdYlGn_r", vmin=0, vmax=2,
            annot=True, fmt=".1f", linewidths=0.5,
            cbar_kws={"label": "0=BAJO, 1=MEDIO, 2=ALTO"})
axes[2].set_title("Riesgo Medio por Día de Semana y Mes")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

plt.tight_layout()
save(fig, "10_target_analisis")

# =============================================================================
# 8. CORRELACIONES
# =============================================================================
print("\n" + "="*70)
print("8. CORRELACIONES Y MULTICOLINEALIDAD")
print("="*70)

feature_cols = [
    "acled_region_n_violentos", "acled_region_n_explosions", "acled_region_fatalities",
    "acled_hormuz_n_violentos", "acled_hormuz_fatalities",
    "gdelt_n_eventos", "gdelt_goldstein_mean", "gdelt_goldstein_min", "gdelt_n_conflicto",
    "gdelt_n_mentions", "gdelt_n_articles",
    "maritime_n_incidentes", "maritime_severidad_max", "maritime_n_ataques_directos",
    "ukmto_n_incidentes", "ukmto_n_attacks",
    "opensky_n_vuelos",
    "nivel_num"
]
feature_cols = [c for c in feature_cols if c in df_int.columns]
corr = df_int[feature_cols].corr()

# Correlación con target
target_corr = corr["nivel_num"].drop("nivel_num").sort_values(ascending=False)
print("\n  Correlaciones con nivel_riesgo (codificado 0/1/2):")
print(target_corr.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Análisis de Correlaciones — Dataset Integrado",
             fontsize=13, fontweight="bold")

# Heatmap completo
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=axes[0], mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, annot=False, linewidths=0.3,
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Matriz de Correlaciones (Pearson)")
axes[0].tick_params(axis="x", rotation=45, labelsize=7)
axes[0].tick_params(axis="y", labelsize=7)

# Correlación con target (barplot)
colors_bar = [PALETTE["ALTO"] if v > 0.1 else PALETTE["BAJO"] if v < -0.1 else "gray"
              for v in target_corr.values]
axes[1].barh(target_corr.index, target_corr.values, color=colors_bar)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].axvline(0.1, color="orange", linestyle="--", linewidth=0.8, label="|r|=0.1")
axes[1].axvline(-0.1, color="orange", linestyle="--", linewidth=0.8)
axes[1].set_title("Correlación de cada Feature con nivel_riesgo")
axes[1].set_xlabel("Pearson r")
axes[1].legend(fontsize=8)
for i, (val, feat) in enumerate(zip(target_corr.values, target_corr.index)):
    axes[1].text(val + 0.005 if val >= 0 else val - 0.005, i,
                 f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

plt.tight_layout()
save(fig, "11_correlaciones")

# =============================================================================
# 9. SERIES DE TIEMPO MULTIFUENTE
# =============================================================================
print("\n" + "="*70)
print("9. SERIES DE TIEMPO MULTIFUENTE")
print("="*70)

fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
fig.suptitle("Series de Tiempo Multifuente — Estrecho de Hormuz 2024",
             fontsize=14, fontweight="bold")

# Fondo de nivel_riesgo
riesgo_color = {"BAJO": "#d5f5e3", "MEDIO": "#fef9e7", "ALTO": "#fadbd8"}
for ax in axes:
    for _, row in df_int.iterrows():
        ax.axvspan(row["fecha"] - pd.Timedelta("0.5D"),
                   row["fecha"] + pd.Timedelta("0.5D"),
                   alpha=0.15, color=riesgo_color[row["nivel_riesgo"]], linewidth=0)

# Panel 1: ACLED Hormuz
axes[0].fill_between(df_int["fecha"], df_int["acled_hormuz_n_violentos"],
                      color="#e74c3c", alpha=0.7, label="Violentos Hormuz")
axes[0].set_ylabel("N° Eventos")
axes[0].set_title("ACLED — Eventos Violentos Directos en Hormuz")
axes[0].legend(loc="upper right", fontsize=8)

# Panel 2: GDELT Goldstein
axes[1].plot(df_int["fecha"], df_int["gdelt_goldstein_mean"],
             color="#3498db", linewidth=1.5, label="Goldstein medio")
axes[1].fill_between(df_int["fecha"], df_int["gdelt_goldstein_mean"], 0,
                      where=df_int["gdelt_goldstein_mean"] < 0,
                      alpha=0.3, color="#e74c3c")
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("Goldstein")
axes[1].set_title("GDELT — Tono Mediático (Goldstein Scale)")
axes[1].legend(loc="upper right", fontsize=8)

# Panel 3: UKMTO ataques
axes[2].bar(df_int["fecha"], df_int["ukmto_n_attacks"],
            color=PALETTE["ALTO"], alpha=0.8, label="UKMTO Attacks", width=1)
axes[2].bar(df_int["fecha"], df_int["ukmto_n_suspicious"],
            bottom=df_int["ukmto_n_attacks"],
            color=PALETTE["MEDIO"], alpha=0.7, label="UKMTO Suspicious", width=1)
axes[2].set_ylabel("N° Incidentes")
axes[2].set_title("UKMTO — Incidentes Oficiales 2024")
axes[2].legend(loc="upper right", fontsize=8)

# Panel 4: Maritime severidad
axes[3].bar(df_int["fecha"], df_int["maritime_severidad_max"],
            color="#9b59b6", alpha=0.8, label="Severidad Máx", width=1)
axes[3].axhline(4, color=PALETTE["ALTO"], linestyle="--", linewidth=1, label="Umbral ALTO (≥4)")
axes[3].axhline(2, color=PALETTE["MEDIO"], linestyle="--", linewidth=1, label="Umbral MEDIO (≥2)")
axes[3].set_ylabel("Severidad")
axes[3].set_title("Maritime Incidents — Severidad Máxima por Día")
axes[3].legend(loc="upper right", fontsize=8)

# Panel 5: nivel_riesgo
nivel_num_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}
colors_pts = [PALETTE[r] for r in df_int["nivel_riesgo"]]
axes[4].scatter(df_int["fecha"], df_int["nivel_riesgo"].map(nivel_num_map),
                c=colors_pts, s=30, zorder=5, edgecolors="white", linewidth=0.3)
axes[4].set_yticks([0, 1, 2])
axes[4].set_yticklabels(["BAJO", "MEDIO", "ALTO"])
axes[4].set_title("Target — nivel_riesgo Diario")
axes[4].set_ylabel("Riesgo")

axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[4].xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(axes[4].xaxis.get_majorticklabels(), rotation=30, ha="right")

# Marcar eventos clave en todos los paneles
for fecha_str, label in eventos_clave.items():
    for ax in axes:
        ax.axvline(pd.to_datetime(fecha_str), color="black", linestyle="--",
                   alpha=0.5, linewidth=1)
axes[0].set_xlim(df_int["fecha"].min(), df_int["fecha"].max())

plt.tight_layout()
save(fig, "12_series_tiempo_multifuente")

# =============================================================================
# 10. RESUMEN DE HALLAZGOS
# =============================================================================
print("\n" + "="*70)
print("10. RESUMEN DE HALLAZGOS EDA")
print("="*70)

print("""
HALLAZGOS CLAVE:
─────────────────────────────────────────────────────────────────────────────
1. TARGET (nivel_riesgo):
   - BAJO: 269 días (73.5%) | MEDIO: 78 días (21.3%) | ALTO: 19 días (5.2%)
   - Fuerte desbalance de clases → necesario SMOTE o class_weight en modelos

2. ACLED REGIÓN:
   - 37 722 eventos en 2024 — Yemen domina (34%), seguido de EE.UU. (30%)
   - Yemen tiene actividad bélica diaria constante (línea base alta)
   - Picos en Enero (Torre 22) y Octubre (respuesta Israel→Irán)

3. ACLED HORMUZ (señal directa):
   - Solo 81 eventos directos en bbox Hormuz — señal escasa pero precisa
   - Actores: fuerzas iraníes y milicias proxies de Irán
   - Concentración en Abr-Oct (escalada post-ataque Irán→Israel)

4. GDELT:
   - Goldstein medio: −0.7 (tono levemente negativo todo el año)
   - Pico de conflicto verbal (QuadClass=3) en Abril (ataque Irán→Israel)
   - Alta correlación NumMentions ↔ NumArticles (multicolinealidad)

5. OPENSKY:
   - Solo 9 fechas disponibles por límite free tier
   - Días de escalada (Abr 13-14) tienen MENOS vuelos → restricciones de espacio
   - Señal útil pero muy escasa para modelos supervisados

6. MARITIME INCIDENTS:
   - 31 incidentes manuales, severidad media 3.0
   - Hormuz: IRGC seizures (alta severidad), Mar Rojo: ataques hutíes continuos
   - Febrero pico (MV Rubymar hundido), Abril máximo (MSC Aries seizure + Iran)

7. UKMTO 2024:
   - 99 incidentes originales en 80 días — fuente más densa y continua
   - Junio el mes más activo (14 incidentes), Enero y Agosto en segundo lugar
   - 34 ataques confirmados + 64 warnings — señal de alta calidad

8. CORRELACIONES CON TARGET:
   - Más correlacionadas: ukmto_n_attacks, maritime_severidad_max,
     acled_hormuz_n_violentos, maritime_n_incidentes
   - GDELT y ACLED_región tienen correlación baja → señal de ruido regional
   - OpenSky sin correlación significativa (solo 9 puntos)

9. MULTICOLINEALIDAD:
   - gdelt_n_mentions ↔ gdelt_n_articles: r ≈ 0.97 → eliminar una
   - acled_region_n_violentos ↔ acled_region_n_explosions: r ≈ 0.89
   - maritime_severidad_max ↔ maritime_n_incidentes: r ≈ 0.82

10. RECOMENDACIONES PARA MODELOS ML:
    - Usar class_weight='balanced' o SMOTE para desbalance BAJO/ALTO
    - Eliminar features con r²<0.01 con target
    - Escalar antes de Logistic Regression y KNN
    - Random Forest/XGBoost más robustos al desbalance
    - Evaluar con F1-macro y Cohen's Kappa (no accuracy — misleading con desbalance)
─────────────────────────────────────────────────────────────────────────────
""")

print(f"\nFiguras guardadas en: {FIGS}")
print(f"Total figuras: {len(list(FIGS.glob('*.png')))}")
