import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import AISSTREAM_API_KEY, HORMUZ_BBOX

load_dotenv()

WS_URL   = "wss://stream.aisstream.io/v0/stream"
DURATION = 30  # segundos de escucha

async def stream_ais(duration=DURATION):
    print(f"Conectando a AISStream — escuchando {duration}s sobre Hormuz...")
    records = []

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({
            "APIKey": AISSTREAM_API_KEY,
            "BoundingBoxes": [[
                [HORMUZ_BBOX["lat_min"], HORMUZ_BBOX["lon_min"]],
                [HORMUZ_BBOX["lat_max"], HORMUZ_BBOX["lon_max"]]
            ]],
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
        }))

        start = asyncio.get_event_loop().time()
        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("MessageType", "")
            meta  = msg.get("Metadata", {})

            if mtype == "PositionReport":
                pos = msg["Message"]["PositionReport"]
                records.append({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "mmsi":        meta.get("MMSI"),
                    "ship_name":   meta.get("ShipName", "").strip(),
                    "latitude":    meta.get("Latitude"),
                    "longitude":   meta.get("Longitude"),
                    "sog":         pos.get("Sog"),       # velocidad nudos
                    "cog":         pos.get("Cog"),       # rumbo
                    "nav_status":  pos.get("NavigationalStatus"),
                    "msg_type":    "PositionReport"
                })

            elif mtype == "ShipStaticData":
                static = msg["Message"]["ShipStaticData"]
                records.append({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "mmsi":        meta.get("MMSI"),
                    "ship_name":   static.get("Name", "").strip(),
                    "latitude":    meta.get("Latitude"),
                    "longitude":   meta.get("Longitude"),
                    "ship_type":   static.get("Type"),
                    "destination": static.get("Destination", "").strip(),
                    "msg_type":    "ShipStaticData"
                })

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= duration:
                print(f"Tiempo completado ({duration}s)")
                break

    return records


if __name__ == "__main__":
    records = asyncio.run(stream_ais(DURATION))

    if records:
        df = pd.DataFrame(records)
        print(f"\nBuques detectados en Hormuz: {len(df)}")
        print(df[["ship_name","latitude","longitude","sog","nav_status"]].dropna(subset=["ship_name"]).head(10).to_string())
        df.to_csv("../../data/raw/raw_aisstream_hormuz.csv", index=False)
        print("Guardado: data/raw/raw_aisstream_hormuz.csv")
    else:
        print("Sin datos AIS recibidos.")
