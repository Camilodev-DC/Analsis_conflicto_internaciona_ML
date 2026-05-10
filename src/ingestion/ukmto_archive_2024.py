"""
Scraper del archivo histórico UKMTO 2024.
URL: https://www.ukmto.org/ukmto-products/warnings/2024
Navega mes a mes y extrae todos los incidentes con fecha, referencia y nombre.
También intenta entrar a cada enlace para obtener el texto completo del informe.
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re
import time
from pathlib import Path

BASE_URL = "https://www.ukmto.org/ukmto-products/warnings/2024"
OUT_RAW  = Path(__file__).parent.parent.parent / "data" / "raw"

MONTHS_2024 = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November",
]


def get_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
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
        pass


def parse_table_text(text: str, month: str) -> list[dict]:
    """
    Parsea el texto de la tabla de incidentes UKMTO.
    Formato: REFERENCIA  FECHA  HORA  NOMBRE
    """
    records = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    row_pat = re.compile(
        r"^(UKMTO\s+INCIDENT\s+[\w/]+)\s+"
        r"(\d{2}/\d{2}/\d{4})\s+"
        r"(\d{2}:\d{2})\s+"
        r"(.+)$"
    )

    for line in lines:
        m = row_pat.match(line)
        if m:
            ref, date_str, time_str, name = m.groups()
            try:
                fecha = pd.to_datetime(date_str, dayfirst=True).strftime("%Y-%m-%d")
            except:
                fecha = date_str
            records.append({
                "referencia": ref.strip(),
                "fecha": fecha,
                "hora": time_str,
                "nombre": name.strip(),
                "mes": month,
            })

    return records


def click_month(driver, month_name: str) -> bool:
    """Hace click en el tab del mes especificado."""
    try:
        # Selector por clase CSS específica del sitio UKMTO
        btns = driver.find_elements(By.CSS_SELECTOR, "button[class*='monthItem']")
        for btn in btns:
            if month_name.lower() in btn.text.lower():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(3)
                return True
        print(f"  Boton '{month_name}' no encontrado entre {len(btns)} botones")
        return False
    except Exception as e:
        print(f"  No se pudo clickear {month_name}: {e}")
        return False


def get_incident_links(driver) -> list[str]:
    """Obtiene los enlaces a incidentes individuales de la tabla."""
    links = []
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tr a, .table tr a, [role='row'] a")
        for r in rows:
            href = r.get_attribute("href")
            if href and "ukmto" in href.lower():
                links.append(href)
    except:
        pass
    return list(set(links))


def scrape_incident_detail(driver, url: str) -> str:
    """Abre un incidente individual y extrae el texto."""
    try:
        driver.get(url)
        time.sleep(3)
        body = driver.find_element(By.TAG_NAME, "body").text
        return body[:2000]
    except:
        return ""


if __name__ == "__main__":
    print("Iniciando scraping archivo UKMTO 2024...")
    driver = get_driver()
    all_records = []

    try:
        driver.get(BASE_URL)
        print(f"Página cargada: {driver.title}")
        time.sleep(5)
        accept_cookies(driver)
        time.sleep(2)

        for month in MONTHS_2024:
            print(f"\n--- {month} ---")
            if not click_month(driver, month):
                # intenta scroll y re-click
                driver.execute_script("window.scrollTo(0, 300)")
                time.sleep(1)
                if not click_month(driver, month):
                    print(f"  Saltando {month}")
                    continue

            time.sleep(3)
            body_text = driver.find_element(By.TAG_NAME, "body").text
            records = parse_table_text(body_text, month)
            print(f"  Incidentes parseados: {len(records)}")

            for r in records[:3]:
                print(f"    {r['referencia']} | {r['fecha']} | {r['nombre'][:50]}")

            all_records.extend(records)
            time.sleep(1)

        # Guardar resultado
        if all_records:
            df = pd.DataFrame(all_records)
            # Filtrar solo reportes originales (sin UPDATE)
            df["es_update"] = df["referencia"].str.contains(r"/\d+$", regex=True)
            df_original = df[~df["es_update"]].copy()

            out = OUT_RAW / "raw_ukmto_2024_archivo.csv"
            df.to_csv(out, index=False)
            print(f"\nTotal registros (con updates): {len(df)}")
            print(f"Incidentes originales: {len(df_original)}")
            print(f"Guardado: {out}")

            # Resumen por mes
            print("\nIncidentes originales por mes:")
            print(df_original["mes"].value_counts().reindex(MONTHS_2024).to_string())
        else:
            print("Sin datos extraídos.")

    finally:
        driver.quit()
        print("Driver cerrado.")
