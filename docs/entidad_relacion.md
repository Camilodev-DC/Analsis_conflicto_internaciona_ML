# Mapa Entidad-Relación — Sistema de Inteligencia Multifuente
## Estrecho de Hormuz · ML1-2026I

---

## ENTIDADES PRINCIPALES

### 1. VENTANA_TEMPORAL
> Unidad de análisis central — cada fila del dataset final
```
ventana_id       PK  (fecha + region, ej: "2024-04-14_hormuz")
fecha            DATE
region           VARCHAR  ("hormuz", "iran_costa", "golfo_persico")
nivel_riesgo     ENUM     (BAJO / MEDIO / ALTO)  ← TARGET del modelo
score_riesgo     FLOAT    (0.0 – 1.0)            ← alternativa regresión
```

---

### 2. EVENTO_CONFLICTO  ← viene de ACLED
```
evento_id        PK
ventana_id       FK → VENTANA_TEMPORAL
fecha            DATE
tipo_evento      VARCHAR  (Battles, Explosions, Strategic dev...)
sub_tipo         VARCHAR
actor1           VARCHAR
actor2           VARCHAR
lat              FLOAT
lon              FLOAT
fatalidades      INT
severidad        INT      (0-3, geo_precision de ACLED)
fuente           VARCHAR  ("ACLED")
```

---

### 3. NOTICIA  ← viene de GDELT + RSS (BBC, Al Jazeera)
```
noticia_id       PK
ventana_id       FK → VENTANA_TEMPORAL
fecha            DATETIME
titulo           TEXT
url              VARCHAR
dominio          VARCHAR
idioma           VARCHAR
tono             FLOAT    (negativo = conflicto, GDELT goldstein scale)
pais_fuente      VARCHAR
lat_mencion      FLOAT    (geolocalización de la mención)
lon_mencion      FLOAT
fuente           VARCHAR  ("GDELT" | "BBC_RSS" | "ALJAZEERA_RSS")
```

---

### 4. VUELO  ← viene de OpenSky
```
vuelo_id         PK
ventana_id       FK → VENTANA_TEMPORAL
icao24           VARCHAR  (identificador único del avión)
callsign         VARCHAR
pais_origen      VARCHAR
lat              FLOAT
lon              FLOAT
altitud_m        FLOAT
velocidad_ms     FLOAT
en_suelo         BOOL
timestamp        DATETIME
fuente           VARCHAR  ("OPENSKY")
```

---

### 5. BUQUE  ← viene de AISStream
```
buque_id         PK
ventana_id       FK → VENTANA_TEMPORAL
mmsi             VARCHAR  (identificador único del buque)
nombre           VARCHAR
lat              FLOAT
lon              FLOAT
velocidad_nudos  FLOAT
rumbo            FLOAT
estado_nav       INT      (0=navegando, 1=fondead, 5=amarrado...)
tipo_buque       INT      (código AIS: 70=carga, 80=tanquero...)
timestamp        DATETIME
fuente           VARCHAR  ("AISSTREAM")
```

---

### 6. HOTSPOT  ← viene de NASA FIRMS
```
hotspot_id       PK
ventana_id       FK → VENTANA_TEMPORAL
lat              FLOAT
lon              FLOAT
brightness_k     FLOAT    (temperatura de brillo)
frp_mw           FLOAT    (Fire Radiative Power — intensidad)
satelite         VARCHAR  ("MODIS" | "VIIRS_NOAA20" | "VIIRS_SNPP")
confianza        VARCHAR  ("low" | "nominal" | "high")
dia_noche        CHAR(1)  ("D" | "N")
fecha            DATE
fuente           VARCHAR  ("NASA_FIRMS")
```

---

### 7. REGION  ← tabla de referencia geográfica
```
region_id        PK
nombre           VARCHAR  ("hormuz", "iran_costa", "golfo_persico", "mar_rojo")
bbox_norte       FLOAT
bbox_sur         FLOAT
bbox_este        FLOAT
bbox_oeste       FLOAT
pais_principal   VARCHAR
descripcion      TEXT
```

---

### 8. FEATURE_VECTOR  ← tabla agregada para el modelo ML
> Esta es la tabla que alimenta directamente a scikit-learn
```
vector_id        PK
ventana_id       FK → VENTANA_TEMPORAL

-- Features de ACLED
acled_n_eventos       INT
acled_fatalidades     INT
acled_tipo_dominante  VARCHAR

-- Features de GDELT
gdelt_n_articulos     INT
gdelt_tono_promedio   FLOAT
gdelt_tono_std        FLOAT
gdelt_n_menciones_geo INT

-- Features de OpenSky
opensky_n_vuelos      INT
opensky_alt_promedio  FLOAT
opensky_vel_promedio  FLOAT

-- Features de AISStream
ais_n_buques          INT
ais_n_tanqueros       INT
ais_vel_promedio      FLOAT
ais_n_fondead         INT      (buques detenidos = señal anómala)

-- Features de NASA FIRMS
firms_n_hotspots      INT
firms_frp_total       FLOAT
firms_frp_max         FLOAT

-- TARGET
nivel_riesgo          ENUM    (BAJO=0 / MEDIO=1 / ALTO=2)
score_riesgo          FLOAT
```

---

## RELACIONES

```
REGION          ||--o{ VENTANA_TEMPORAL    : "contiene"
VENTANA_TEMPORAL ||--o{ EVENTO_CONFLICTO  : "agrega"
VENTANA_TEMPORAL ||--o{ NOTICIA           : "agrega"
VENTANA_TEMPORAL ||--o{ VUELO             : "agrega"
VENTANA_TEMPORAL ||--o{ BUQUE             : "agrega"
VENTANA_TEMPORAL ||--o{ HOTSPOT           : "agrega"
VENTANA_TEMPORAL ||--|| FEATURE_VECTOR    : "resume en"
```

---

## DIAGRAMA EN TEXTO

```
┌─────────────┐
│   REGION    │
│─────────────│
│ region_id PK│
│ nombre      │
│ bbox_*      │
└──────┬──────┘
       │ 1
       │ contiene
       │ N
┌──────▼──────────────────────────────────────────┐
│              VENTANA_TEMPORAL                    │
│─────────────────────────────────────────────────│
│ ventana_id PK  (fecha + region)                 │
│ fecha                                           │
│ region                                          │
│ nivel_riesgo  ← TARGET                          │
│ score_riesgo  ← TARGET alternativo              │
└──┬──────┬──────┬──────┬──────┬──────────────────┘
   │      │      │      │      │
   │1     │1     │1     │1     │1
   │N     │N     │N     │N     │N
   ▼      ▼      ▼      ▼      ▼
┌──────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌────────┐
│EVENTO│ │NOTI-│ │VUELO│ │BUQUE │ │HOT-    │
│CONF- │ │CIA  │ │     │ │      │ │SPOT    │
│LICTO │ │     │ │     │ │      │ │        │
│──────│ │─────│ │─────│ │──────│ │────────│
│ACLED │ │GDELT│ │OPEN-│ │AIS-  │ │NASA    │
│      │ │RSS  │ │SKY  │ │STREAM│ │FIRMS   │
└──────┘ └─────┘ └─────┘ └──────┘ └────────┘
   │          │      │       │         │
   └──────────┴──────┴───────┴─────────┘
                     │ agrega y resume
                     ▼
         ┌───────────────────────┐
         │    FEATURE_VECTOR     │
         │───────────────────────│
         │ acled_n_eventos       │
         │ acled_fatalidades     │
         │ gdelt_n_articulos     │──► scikit-learn
         │ gdelt_tono_promedio   │    KNN / NaiveBayes
         │ opensky_n_vuelos      │    LogisticRegression
         │ ais_n_buques          │    RandomForest
         │ ais_n_tanqueros       │
         │ firms_n_hotspots      │
         │ firms_frp_total       │
         │ nivel_riesgo TARGET   │
         └───────────────────────┘
```

---

## FLUJO DE CONSTRUCCIÓN DEL DATASET

```
ACLED API      → raw_acled.csv    ─┐
GDELT API      → raw_gdelt.csv    ─┤
OpenSky API    → raw_opensky.csv  ─┤→ JOIN por (fecha + bbox) → FEATURE_VECTOR → modelo
AISStream WS   → raw_ais.csv      ─┤
NASA FIRMS API → raw_firms.csv    ─┘
```

### Clave de JOIN
```python
# Toda fuente se agrupa así antes del join:
df.groupby(['fecha', 'region']).agg({...}).reset_index()

# Luego merge secuencial:
df_final = (acled_agg
    .merge(gdelt_agg,   on=['fecha','region'], how='left')
    .merge(opensky_agg, on=['fecha','region'], how='left')
    .merge(ais_agg,     on=['fecha','region'], how='left')
    .merge(firms_agg,   on=['fecha','region'], how='left')
)
```

---

## BBOX DEL ESTRECHO DE HORMUZ

```python
HORMUZ_BBOX = {
    "lat_min": 25.0,
    "lat_max": 27.5,
    "lon_min": 55.5,
    "lon_max": 58.0
}
```
Todas las fuentes se filtran con este bounding box antes de agregar.
