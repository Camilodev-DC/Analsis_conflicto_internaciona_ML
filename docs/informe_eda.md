# Informe EDA — Estrecho de Hormuz 2024
## Sistema de Inteligencia Multifuente para Predicción de Riesgo Operativo

> **Pregunta de investigación:** ¿Es posible predecir el nivel de riesgo operativo diario en el Estrecho de Hormuz usando tráfico marítimo (UKMTO/AIS), actividad aérea (OpenSky), cobertura mediática (GDELT) y eventos de conflicto (ACLED)?

---

## Ficha del Dataset

| | |
|---|---|
| **Observaciones** | 366 días (2024-01-01 → 2024-12-31) |
| **Features** | 37 variables + 1 target |
| **Fuentes integradas** | 5 (ACLED, GDELT, OpenSky, Maritime Manual, UKMTO) |
| **Valores nulos** | **0** — dataset completamente limpio |
| **Target** | `nivel_riesgo` = BAJO / MEDIO / ALTO |

---

## Distribución del Target

```
BAJO  ████████████████████████████████████████  146 días  (39.9%)
MEDIO ████████████████████████████████████████████████████  201 días  (54.9%)
ALTO  █████  19 días   (5.2%)
```

Desbalance BAJO/ALTO = **7.7x** → requiere `class_weight='balanced'` en todos los modelos

---

## Lo bueno del dataset ✅

### 1. Completitud perfecta
- **Cero valores nulos** en las 366 filas × 38 columnas
- Cobertura temporal continua sin huecos — todos los días del año 2024
- El pipeline de integración rellenó correctamente con 0 los días sin datos de cada fuente

### 2. GDELT — la fuente más poderosa
Las features de cobertura mediática son las **más correlacionadas con el target**:

| Feature | Correlación con nivel_riesgo |
|---|---|
| `gdelt_n_eventos` | **+0.514** |
| `gdelt_goldstein_min` | **−0.507** |
| `gdelt_n_conflicto` | **+0.496** |
| `gdelt_goldstein_mean` | **−0.320** |

- Goldstein Scale media = **−1.79** — coherente con un año de conflicto activo
- El tono mediático se degrada claramente los días de escalada (Abril: ataque Irán→Israel, Octubre: respuesta Israel→Irán)
- `gdelt_goldstein_min` captura el peor evento del día — señal de alarma muy efectiva

### 3. UKMTO 2024 — fuente marítima oficial más densa
- **80 días** con incidentes registrados (vs 29 del dataset manual)
- Datos verificados por autoridad marítima del Reino Unido
- `ukmto_n_incidentes` correlación **r = +0.411** con el target
- 34 ataques confirmados + 64 advertencias — espectro completo de severidad
- Junio fue el mes más activo (14 incidentes), coherente con la escalada del conflicto

### 4. Maritime Incidents — alta calidad semántica
- Severidad calibrada con criterio experto (escala 1–5)
- `maritime_severidad_max`: correlación **r = +0.453** — tercera más alta del dataset
- Captura exactamente los eventos más relevantes del año:
  - MV Rubymar hundido (Feb) — primer buque hundido por hutíes
  - MSC Aries confiscado por IRGC (Abr) — severidad 5 en Hormuz
  - MV Sounion incendiado (Ago) — riesgo de derrame masivo

### 5. Moving Averages MA7 bien construidas
- 6 features MA7 construidas con `shift(1)` — **sin data leakage**
- Capturan tendencias de 7 días previos al día de predicción
- `gdelt_n_conflicto_ma7`: correlación **r = +0.430** — más alta que la versión diaria en algunos modelos

### 6. Complementariedad entre fuentes
- Cuando GDELT tiene señal alta, UKMTO y Maritime pueden estar en 0 y viceversa
- Ninguna fuente domina completamente — el modelo necesita todas para discriminar bien
- ACLED cubre la presión geopolítica regional cuando no hay incidentes directos en Hormuz

### 7. Variabilidad temporal real
- Los 3 niveles de riesgo tienen representación: BAJO, MEDIO y ALTO ocurren en distintos meses
- Abril y Octubre son claramente los meses de mayor riesgo (picos de escalada Irán–Israel)
- Esta variabilidad hace el problema aprendible para los modelos

---

## Lo malo del dataset ❌

### 1. OpenSky — prácticamente inutilizable
- **97.5% de ceros** — solo 9 de 366 días tienen datos de tráfico aéreo
- `opensky_n_grounded` = 0 en el **100% de los días** → varianza cero → columna muerta
- Todas las variables OpenSky están correlacionadas perfectamente entre sí (r > 0.997) → 4 columnas que aportan exactamente la misma información
- Los 9 "outliers" detectados son simplemente los 9 días con datos reales — el artefacto, no la excepción

> **Decisión:** Excluir todas las variables OpenSky del entrenamiento ML.

### 2. Multicolinealidad severa en GDELT y Maritime
9 pares de variables con r > 0.94 — inflando dimensionalidad sin información nueva:

| Par | r | Qué eliminar |
|---|---|---|
| `acled_hormuz_fatalities` ↔ `acled_hormuz_n_violentos` | 1.000 | `fatalities` |
| `opensky_n_airborne` ↔ `opensky_n_vuelos` | 1.000 | `airborne` |
| `gdelt_n_articles` ↔ `gdelt_n_eventos` | 0.984 | `n_articles` |
| `gdelt_n_mentions` ↔ `gdelt_n_eventos` | 0.973 | `n_mentions` |
| `maritime_severidad_sum` ↔ `maritime_severidad_max` | 0.958 | `severidad_sum` |
| `gdelt_n_conflicto` ↔ `gdelt_n_articles` | 0.954 | `n_articles` |
| `maritime_severidad_sum` ↔ `maritime_n_incidentes` | 0.952 | `severidad_sum` |
| `dia_del_año` ↔ `mes` | 0.997 | `dia_del_año` |
| `maritime_n_ataques_directos` ↔ `maritime_n_incidentes` | 0.873 | `n_ataques_directos` |

### 3. ACLED-Hormuz captura protestas, no operaciones militares
- El bbox del Estrecho de Hormuz coincide con el norte de Irán donde ocurren protestas ciudadanas frecuentes
- **73 de 81 eventos** (90%) son protestas de civiles iraníes — no incidentes de seguridad marítima
- `acled_hormuz_n_violentos`: **99.5% de ceros** — solo 2 días con eventos violentos reales en Hormuz
- Correlación real con el target apenas r = 0.17 — señal débil y ruidosa

### 4. GDELT ausente los últimos 62 días
- No hay datos GDELT para Noviembre y Diciembre 2024
- Esos días tienen `gdelt_goldstein_mean = 0` por imputación — valor artificialmente neutro
- Esto sesga el modelo: los últimos 2 meses parecen "tranquilos" aunque podrían no serlo
- Afecta especialmente si se usa validación temporal (Oct–Dic como test set)

### 5. Desbalance de clases 7.7x
- ALTO solo tiene **19 observaciones** — insuficiente para que modelos complejos aprendan la clase
- Un clasificador naive que prediga siempre MEDIO tiene **54.9% de accuracy**
- Accuracy es una métrica completamente engañosa con este dataset
- SMOTE puede generar observaciones sintéticas ALTO pero con riesgo de overfitting

### 6. Target con riesgo de tautología
- El target `nivel_riesgo` fue construido **con reglas basadas en las mismas features** del dataset
- Random Forest puede memorizar las reglas y reportar F1 = 1.0 en entrenamiento
- Solución obligatoria: **validación temporal** (no split aleatorio)

### 7. Columnas de varianza cero o casi-cero
| Variable | % ceros | Problema |
|---|---|---|
| `opensky_n_grounded` | 100% | Varianza cero — eliminar |
| `ukmto_n_hijack` | 100% | Varianza cero — sin secuestros en 2024 |
| `ukmto_n_suspicious` | 99.7% | Quasi-constante — eliminar |
| `acled_hormuz_n_violentos` | 99.5% | Mejor binarizar (0/1) |
| `maritime_n_hormuz` | 99.2% | Mejor binarizar (0/1) |

### 8. Skewness extrema en 9 variables
Variables con skewness > 3.5 afectan directamente la Regresión Logística y KNN:

| Variable | Skewness | Acción |
|---|---|---|
| `ukmto_n_suspicious` | 19.1 | Eliminar |
| `acled_hormuz_n_violentos` | 13.5 | Binarizar |
| `maritime_n_hormuz` | 11.0 | Binarizar |
| `gdelt_n_conflicto` | 4.8 | log1p |
| `gdelt_n_eventos` | 3.4 | log1p |
| `ukmto_n_attacks` | 3.9 | log1p |

---

## Correlaciones clave con el Target

```
gdelt_n_eventos          +0.514  ████████████████████
gdelt_goldstein_min      -0.507  ████████████████████  (negativo = más conflicto)
gdelt_n_conflicto        +0.496  ███████████████████
gdelt_n_articulos        +0.499  ███████████████████
maritime_severidad_max   +0.453  ████████████████
gdelt_n_conflicto_ma7    +0.430  ████████████████
maritime_n_incidentes    +0.412  ███████████████
ukmto_n_incidentes       +0.411  ███████████████
gdelt_goldstein_mean     -0.320  ████████████
ukmto_n_attacks          +0.324  ████████████
acled_region_fatalities  -0.045  ██  (señal débil, ruido regional)
```

---

## Recomendaciones para el Modelo ML

### Features a eliminar (9 columnas)
```python
ELIMINAR = [
    'opensky_n_grounded',       # varianza cero
    'ukmto_n_hijack',           # varianza cero
    'ukmto_n_suspicious',       # 99.7% ceros
    'opensky_n_airborne',       # idéntica a opensky_n_vuelos
    'gdelt_n_articles',         # r=0.98 con gdelt_n_eventos
    'gdelt_n_mentions',         # r=0.97 con gdelt_n_eventos
    'maritime_severidad_sum',   # r=0.96 con maritime_severidad_max
    'acled_hormuz_fatalities',  # r=1.00 con acled_hormuz_n_violentos
    'dia_del_año',              # r=0.997 con mes
]
```

### Transformaciones (log1p para reducir skewness)
```python
TRANSFORMAR = [
    'gdelt_n_conflicto',   # skew 4.85 → log1p
    'gdelt_n_eventos',     # skew 3.43 → log1p
    'ukmto_n_attacks',     # skew 3.85 → log1p
]
BINARIZAR = [
    'acled_hormuz_n_violentos',  # 99.5% ceros
    'maritime_n_hormuz',          # 99.2% ceros
]
```

### Configuración de modelos
| Modelo | Configuración esencial |
|---|---|
| Logistic Regression | `class_weight='balanced'` + StandardScaler |
| KNN | StandardScaler obligatorio + `weights='distance'` |
| Naive Bayes | GaussianNB para continuas |
| Random Forest | `class_weight='balanced'` + `n_estimators=200` |

### Esquema de validación (NO usar split aleatorio)
```
Entrenamiento:  2024-01-01 → 2024-09-30   (273 días)
Test:           2024-10-01 → 2024-12-31   (92 días)
```

### Métricas de evaluación
| Métrica | Por qué usarla |
|---|---|
| **F1-Macro** | Trata todas las clases por igual |
| **F1-ALTO** | Crítica operacionalmente — detectar días de máximo riesgo |
| **Cohen's Kappa** | Ajusta por azar, robusto al desbalance |
| ~~Accuracy~~ | ❌ Misleading — un modelo que siempre prediga MEDIO tiene 54.9% |

---

*Dataset: `data/processed/dataset_integrado.csv` | Figuras: `docs/eda_figures/` | 12 visualizaciones generadas*
