"""
Scraper de incidentes maritimos en el Estrecho de Hormuz / Mar Rojo
Fuentes: Wikipedia + Naval News (accesibles publicamente)
Como alternativa a UKMTO que bloquea scraping externo.
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ─── FUENTE 1: Wikipedia — Ataques hutíes en el Mar Rojo 2023-2024 ────────────
def scrape_wikipedia_houthi():
    url = "https://en.wikipedia.org/wiki/Houthi_attacks_on_shipping_(2023%E2%80%93present)"
    print(f"Scrapeando Wikipedia — ataques hutíes...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Error {r.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    print(f"Tablas encontradas: {len(tables)}")

    frames = []
    for i, table in enumerate(tables):
        try:
            df = pd.read_html(str(table))[0]
            df["tabla_idx"] = i
            frames.append(df)
            print(f"  Tabla {i}: {len(df)} filas, cols: {list(df.columns)}")
        except Exception as e:
            print(f"  Tabla {i} error: {e}")

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all["source"] = "wikipedia_houthi"
        return df_all
    return pd.DataFrame()


# ─── FUENTE 2: Wikipedia — Conflicto Irán-Israel 2024 ────────────────────────
def scrape_wikipedia_iran_israel():
    url = "https://en.wikipedia.org/wiki/Iran%E2%80%93Israel_conflict_(2024)"
    print(f"Scrapeando Wikipedia — conflicto Irán-Israel 2024...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Error {r.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    print(f"Tablas encontradas: {len(tables)}")

    frames = []
    for i, table in enumerate(tables):
        try:
            df = pd.read_html(str(table))[0]
            df["tabla_idx"] = i
            frames.append(df)
            print(f"  Tabla {i}: {len(df)} filas")
        except Exception as e:
            print(f"  Tabla {i} error: {e}")

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all["source"] = "wikipedia_iran_israel"
        return df_all
    return pd.DataFrame()


# ─── FUENTE 3: Naval News — noticias maritimas del Golfo ─────────────────────
def scrape_naval_news():
    url = "https://www.navalnews.com/naval-news/2024/"
    print(f"Scrapeando Naval News 2024...")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"Error {r.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.find_all("article")
    print(f"Artículos encontrados: {len(articles)}")

    records = []
    keywords = ["hormuz", "iran", "houthi", "red sea", "gulf", "strait", "vessel", "ship"]
    for art in articles:
        title = art.find("h2") or art.find("h3")
        date  = art.find("time")
        link  = art.find("a", href=True)

        if not title:
            continue

        title_text = title.get_text(strip=True).lower()
        if any(kw in title_text for kw in keywords):
            records.append({
                "fecha":   date["datetime"][:10] if date and date.get("datetime") else None,
                "titulo":  title.get_text(strip=True),
                "url":     link["href"] if link else None,
                "source":  "naval_news"
            })

    print(f"  Artículos relevantes: {len(records)}")
    return pd.DataFrame(records)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pip_check = True
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Instalando beautifulsoup4...")
        import subprocess
        subprocess.run(["pip", "install", "beautifulsoup4", "lxml", "-q"])
        from bs4 import BeautifulSoup

    all_data = {}

    # 1. Wikipedia Houthis
    df_houthi = scrape_wikipedia_houthi()
    if not df_houthi.empty:
        df_houthi.to_csv("../../data/raw/raw_wikipedia_houthi.csv", index=False)
        print(f"Guardado: raw_wikipedia_houthi.csv ({len(df_houthi)} filas)\n")
        all_data["houthi"] = df_houthi

    time.sleep(1)

    # 2. Wikipedia Iran-Israel
    df_iran = scrape_wikipedia_iran_israel()
    if not df_iran.empty:
        df_iran.to_csv("../../data/raw/raw_wikipedia_iran_israel.csv", index=False)
        print(f"Guardado: raw_wikipedia_iran_israel.csv ({len(df_iran)} filas)\n")
        all_data["iran_israel"] = df_iran

    time.sleep(1)

    # 3. Naval News
    df_naval = scrape_naval_news()
    if not df_naval.empty:
        df_naval.to_csv("../../data/raw/raw_naval_news.csv", index=False)
        print(f"Guardado: raw_naval_news.csv ({len(df_naval)} filas)\n")
        all_data["naval"] = df_naval

    print("\n=== RESUMEN ===")
    for k, df in all_data.items():
        print(f"{k}: {len(df)} registros")
