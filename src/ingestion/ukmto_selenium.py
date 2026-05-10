"""
Scraper UKMTO v2 — extrae incidentes del mapa interactivo con Selenium
Los datos están en un mapa ArcGIS, se extraen aceptando cookies y
esperando que cargue el contenido dinámico.
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import re

def get_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def accept_cookies(driver):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(text(),'ACCEPT') or contains(text(),'Accept')]"))
        )
        btn.click()
        time.sleep(1)
        print("Cookies aceptadas.")
    except:
        print("No se encontró botón de cookies.")


def get_full_body_text(driver):
    """Extrae todo el texto visible de la página después de carga dinámica"""
    time.sleep(5)
    return driver.find_element(By.TAG_NAME, "body").text


def parse_incidents_from_text(text):
    """Parsea incidentes del texto completo de la página"""
    records = []
    lines = text.split("\n")

    i = 0
    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # detectar tipo de reporte
        if any(kw in line for kw in ["Advisory", "Warning", "Incident"]) and re.search(r"UKMTO #\d+|\d{4}-\d+", line):
            if current:
                records.append(current)
            current = {"titulo": line, "tipo": ""}
            if "Advisory" in line:
                current["tipo"] = "Advisory"
            elif "Warning" in line:
                current["tipo"] = "Warning"
            else:
                current["tipo"] = "Incident"

        # detectar fecha
        elif re.match(r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", line):
            if current:
                current["fecha"] = line

        # detectar coordenadas
        elif re.search(r"\d+°\d+'\s*[NS].*\d+°\d+'\s*[EW]", line):
            if current:
                current["coordenadas"] = line

        # descripción
        elif current and "titulo" in current and "descripcion" not in current and len(line) > 30:
            current["descripcion"] = line

    if current:
        records.append(current)

    return records


def scroll_and_extract(driver):
    """Scroll para cargar todos los elementos de la lista"""
    records_text = []

    # intentar encontrar lista de incidentes
    selectors = [
        ".incident-list", ".results", ".list-item",
        "[class*='incident']", "[class*='report']",
        ".esri-widget", ".calcite-list"
    ]

    for sel in selectors:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            print(f"Encontrado selector '{sel}': {len(els)} elementos")
            for el in els[:5]:
                txt = el.text.strip()
                if txt:
                    records_text.append(txt)
                    print(f"  >> {txt[:100]}")

    return records_text


ARCHIVE_URLS = [
    # Possible historical archive pages — UKMTO products/warnings by year
    ("https://www.ukmto.org/ukmto-products/warnings/2024", "archive_2024"),
    ("https://www.ukmto.org/ukmto-products/warnings",      "archive_warnings"),
    ("https://www.ukmto.org/indian-ocean/incidents/2024",  "archive_io_2024"),
    ("https://www.ukmto.org/recent-incidents",             "recent"),
]


def scrape_url(driver, url, label):
    print(f"\n[{label}] Cargando: {url}")
    driver.get(url)
    accept_cookies(driver)
    time.sleep(8)
    text = get_full_body_text(driver)
    print(f"  Texto: {len(text)} chars")
    return text


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    print("Iniciando UKMTO scraper con búsqueda de archivo 2024...")
    driver = get_driver()
    all_records = []

    try:
        for url, label in ARCHIVE_URLS:
            try:
                text = scrape_url(driver, url, label)
                out_file = f"../../data/raw/ukmto_raw_{label}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"  Texto guardado: {out_file}")

                records = parse_incidents_from_text(text)
                print(f"  Incidentes parseados: {len(records)}")
                for r in records:
                    r["source_label"] = label
                all_records.extend(records)

            except Exception as e:
                print(f"  Error en {url}: {e}")

        if all_records:
            df = pd.DataFrame(all_records)
            df = df.drop_duplicates(subset=["titulo"])
            df.to_csv("../../data/raw/raw_ukmto.csv", index=False)
            print(f"\nTotal incidentes unicos: {len(df)}")
            print(f"Guardado: data/raw/raw_ukmto.csv")
        else:
            print("Sin incidentes extraidos.")

    finally:
        driver.quit()
        print("Driver cerrado.")
