"""
K-Means para agrupar los 12 meses de 2024 segun intensidad del conflicto.
Cada punto = un mes, features = promedios mensuales de las variables clave.
Objetivo: descubrir cuantos "regimenes" de conflicto existieron en 2024.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.preparar_dataset import preparar

FIGS = Path(__file__).parent.parent.parent / "docs" / "model_figures"
FIGS.mkdir(parents=True, exist_ok=True)

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

# Features representativas del conflicto para clustering mensual.
# acled_region_fatalities excluida: varianza entre centroides=0.035 (no discrimina).
FEATURES_CLUSTER = [
    "gdelt_n_eventos",
    "gdelt_goldstein_mean",
    "gdelt_goldstein_min",
    "gdelt_n_conflicto",
    "ukmto_n_incidentes",
    "ukmto_n_attacks",
    "maritime_n_incidentes",
    "maritime_severidad_max",
    "acled_region_n_explosions",
]


def construir_perfil_mensual(df_full):
    """Agrega el dataset completo a nivel mensual con promedios."""
    df = df_full.copy()
    df["mes_num"] = df["fecha"].dt.month

    # Filtrar solo las features disponibles
    features_ok = [f for f in FEATURES_CLUSTER if f in df.columns]

    agg = df.groupby("mes_num")[features_ok].mean().reset_index()
    agg["mes_nombre"] = agg["mes_num"].map(MESES_ES)

    # Añadir % días ALTO por mes
    nivel_map = {"BAJO": 0, "MEDIO": 1, "ALTO": 2}
    df["nivel_num"] = df["nivel_riesgo"].map(nivel_map)
    alto_pct = df.groupby("mes_num")["nivel_num"].apply(
        lambda x: (x == 2).mean() * 100
    ).reset_index(name="pct_alto")
    medio_pct = df.groupby("mes_num")["nivel_num"].apply(
        lambda x: (x == 1).mean() * 100
    ).reset_index(name="pct_medio")

    agg = agg.merge(alto_pct, on="mes_num").merge(medio_pct, on="mes_num")
    return agg, features_ok


def elbow_silhouette(X_sc, max_k=6):
    """Calcula inercia y silhouette para k=2..max_k."""
    ks = range(2, max_k + 1)
    inercias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sc, labels))
    return list(ks), inercias, silhouettes


def plot_elbow(ks, inercias, silhouettes, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ks, inercias, "o-", color="steelblue", lw=2)
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.set_title("Método del Codo")
    ax1.set_xticks(ks)

    ax2.plot(ks, silhouettes, "o-", color="darkorange", lw=2)
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette por k")
    ax2.set_xticks(ks)

    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def plot_clusters(agg, labels, k, ruta):
    """Scatter: gdelt_n_conflicto vs maritime_severidad_max, coloreado por cluster."""
    colores_cluster = ["steelblue", "darkorange", "crimson", "seagreen"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for c in range(k):
        mask = labels == c
        ax.scatter(
            agg.loc[mask, "gdelt_n_conflicto"],
            agg.loc[mask, "maritime_severidad_max"],
            s=120, color=colores_cluster[c],
            label=f"Cluster {c}",
            zorder=3,
        )
        for _, row in agg[mask].iterrows():
            ax.annotate(
                row["mes_nombre"],
                (row["gdelt_n_conflicto"], row["maritime_severidad_max"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9,
            )

    ax.set_xlabel("Eventos de conflicto GDELT (media mensual, log1p)")
    ax.set_ylabel("Severidad marítima máxima (media mensual)")
    ax.set_title(f"Clustering K-Means ({k} clusters) — Meses 2024")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def plot_heatmap(agg, labels, features_ok, ruta):
    """Heatmap: cluster × feature con valor z-score."""
    agg_c = agg.copy()
    agg_c["cluster"] = labels
    perfil = agg_c.groupby("cluster")[features_ok].mean()

    # Z-score por columna para comparar escalas
    perfil_z = (perfil - perfil.mean()) / (perfil.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(3, len(perfil) * 1.2)))
    im = ax.imshow(perfil_z.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(features_ok)))
    ax.set_xticklabels(features_ok, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(perfil)))

    meses_por_cluster = agg_c.groupby("cluster")["mes_nombre"].apply(
        lambda x: ", ".join(x)
    )
    ax.set_yticklabels(
        [f"C{i}: {meses_por_cluster.get(i, '')}" for i in perfil.index],
        fontsize=9,
    )
    ax.set_title("Perfil de clusters — Z-score por feature")
    plt.colorbar(im, ax=ax, label="Z-score")
    plt.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def plot_barras_riesgo(agg, labels, k, ruta):
    """Barras: % días ALTO y MEDIO por cluster."""
    agg_c = agg.copy()
    agg_c["cluster"] = labels
    perfil = agg_c.groupby("cluster")[["pct_alto", "pct_medio"]].mean()

    x = np.arange(k)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, perfil["pct_alto"],  w, label="% días ALTO",  color="crimson")
    ax.bar(x + w/2, perfil["pct_medio"], w, label="% días MEDIO", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {i}" for i in perfil.index])
    ax.set_ylabel("% días")
    ax.set_title("Distribución de riesgo por cluster")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {ruta.name}")


def main():
    print("=" * 55)
    print("  K-MEANS — Clustering mensual de intensidad 2024")
    print("=" * 55)

    data = preparar(verbose=False)
    df_full = data["df_full"]

    # ── Construir perfil mensual ──────────────────────────────────────────────
    agg, features_ok = construir_perfil_mensual(df_full)
    print(f"\nFeatures usadas ({len(features_ok)}): {features_ok}")
    print(f"\nPerfil mensual:\n{agg[['mes_nombre'] + features_ok[:4] + ['pct_alto']].to_string(index=False)}")

    # ── Escalar (12 puntos) ───────────────────────────────────────────────────
    X = agg[features_ok].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # ── Elbow + Silhouette ────────────────────────────────────────────────────
    ks, inercias, silhouettes = elbow_silhouette(X_sc, max_k=5)
    plot_elbow(ks, inercias, silhouettes, FIGS / "kmeans_elbow.png")

    print("\nMétricas por k:")
    print(f"{'k':>4} {'Inercia':>10} {'Silhouette':>12}")
    for k, ine, sil in zip(ks, inercias, silhouettes):
        print(f"{k:>4} {ine:>10.2f} {sil:>12.3f}")

    # ── k óptimo: mejor silhouette ────────────────────────────────────────────
    k_opt = ks[np.argmax(silhouettes)]
    print(f"\nk óptimo por Silhouette: {k_opt}")

    # ── Modelo final ──────────────────────────────────────────────────────────
    km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    agg["cluster"] = labels

    db = davies_bouldin_score(X_sc, labels)
    sil_final = silhouette_score(X_sc, labels)
    print(f"Silhouette final : {sil_final:.3f}  (>0.5 = bueno)")
    print(f"Davies-Bouldin   : {db:.3f}  (<1.0 = bueno)")

    # ── Resultados ────────────────────────────────────────────────────────────
    print("\nAsignación de clusters:")
    print(f"{'Mes':<6} {'Cluster':>8} {'%ALTO':>8} {'%MEDIO':>8}")
    for _, row in agg.sort_values("mes_num").iterrows():
        print(f"{row['mes_nombre']:<6} {int(row['cluster']):>8} "
              f"{row['pct_alto']:>8.1f} {row['pct_medio']:>8.1f}")

    meses_por_cluster = agg.groupby("cluster")["mes_nombre"].apply(
        lambda x: ", ".join(x)
    )
    print("\nClusters:")
    for c, meses in meses_por_cluster.items():
        pct_a = agg[agg["cluster"] == c]["pct_alto"].mean()
        pct_m = agg[agg["cluster"] == c]["pct_medio"].mean()
        print(f"  Cluster {c}: {meses}  (ALTO={pct_a:.1f}%, MEDIO={pct_m:.1f}%)")

    # ── Figuras ───────────────────────────────────────────────────────────────
    plot_clusters(agg, labels, k_opt, FIGS / "kmeans_scatter.png")
    plot_heatmap(agg, labels, features_ok, FIGS / "kmeans_heatmap.png")
    plot_barras_riesgo(agg, labels, k_opt, FIGS / "kmeans_riesgo.png")

    return km, agg, labels


if __name__ == "__main__":
    main()
