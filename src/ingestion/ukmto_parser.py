"""
Parser para el texto completo de UKMTO extraído con Selenium.
Extrae todos los incidentes del texto estructurado en bloques.
También intenta scraping de páginas de archivo histórico 2024.
"""
import re
import pandas as pd
from pathlib import Path

MONTHS = {
    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}

def parse_ukmto_text(text: str) -> list[dict]:
    """
    Extrae incidentes del texto UKMTO.
    Formato esperado:
        <Tipo> UKMTO #<N>
        <D> <Month> <YYYY>
        <descripción...>
    """
    records = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    header_pat = re.compile(
        r"^(Advisory|Attack|Suspicious Activity|Hijack|Warning)\s+UKMTO\s+#(\d+)$",
        re.IGNORECASE
    )
    date_pat = re.compile(
        r"^(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})$",
        re.IGNORECASE
    )

    i = 0
    while i < len(lines):
        m = header_pat.match(lines[i])
        if m:
            tipo = m.group(1)
            numero = int(m.group(2))

            # next non-empty line should be date
            fecha_str = None
            fecha_raw = None
            desc_start = i + 1
            if i + 1 < len(lines):
                dm = date_pat.match(lines[i + 1])
                if dm:
                    day, month_name, year = int(dm.group(1)), dm.group(2).capitalize(), int(dm.group(3))
                    month = MONTHS.get(month_name, 0)
                    fecha_str = f"{year}-{month:02d}-{day:02d}"
                    fecha_raw = lines[i + 1]
                    desc_start = i + 2

            # gather description lines until next header or date
            desc_lines = []
            j = desc_start
            while j < len(lines) and not header_pat.match(lines[j]):
                desc_lines.append(lines[j])
                j += 1

            descripcion = " ".join(desc_lines).strip()

            # extract location hint
            location = None
            loc_m = re.search(r"\d+NM\s+\w+\s+of\s+([^,.]+)", descripcion)
            if loc_m:
                location = loc_m.group(0).strip()

            records.append({
                "ukmto_id": numero,
                "tipo": tipo,
                "fecha": fecha_str,
                "fecha_raw": fecha_raw,
                "descripcion": descripcion[:500],
                "location_hint": location,
                "year": int(fecha_str[:4]) if fecha_str else None,
            })
            i = j
        else:
            i += 1

    return records


def try_parse_file(path: str) -> pd.DataFrame:
    text = Path(path).read_text(encoding="utf-8")
    records = parse_ukmto_text(text)
    print(f"Incidentes parseados de {path}: {len(records)}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    raw_path = "../../data/raw/ukmto_raw_text.txt"
    df = try_parse_file(raw_path)

    if df.empty:
        print("No se parsearon incidentes.")
    else:
        print(f"\nResumen:")
        print(f"  Total: {len(df)}")
        print(f"  Por tipo:\n{df['tipo'].value_counts().to_string()}")
        print(f"  Por año:\n{df['year'].value_counts().sort_index().to_string()}")
        print(f"\nPrimeros 5:")
        for _, r in df.head(5).iterrows():
            print(f"  #{r['ukmto_id']} [{r['tipo']}] {r['fecha']} — {r['descripcion'][:80]}")

        out = "../../data/raw/raw_ukmto_parsed.csv"
        df.to_csv(out, index=False)
        print(f"\nGuardado: {out}")
