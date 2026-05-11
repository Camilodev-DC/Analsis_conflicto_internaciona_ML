# Informe de Modelos ML — Estrecho de Hormuz 2024
## Naive Bayes y K-Means: Clasificación y Clustering de Riesgo Operativo

> **Pregunta de investigación:** ¿Es posible predecir el nivel de riesgo operativo diario en el Estrecho de Hormuz usando tráfico marítimo (UKMTO), cobertura mediática (GDELT) y eventos de conflicto (ACLED)?

---

## 1. Configuración del Dataset

| Parámetro | Valor |
|---|---|
| Observaciones totales | 366 días (2024-01-01 → 2024-12-31) |
| Features tras limpieza | 24 variables |
| Split temporal | 60 / 20 / 20 (temporal, sin shuffle) |
| Train | 219 días (Ene–Ago 06) |
| Validación | 73 días (Ago 07–Oct 18) |
| Test | 74 días (Oct 19–Dic 31) |

### Distribución del target por split

| Clase | Train | Val | Test |
|---|---|---|---|
| BAJO | 71 (32.4%) | 13 (17.8%) | 62 (83.8%) |
| MEDIO | 134 (61.2%) | 55 (75.3%) | 12 (16.2%) |
| ALTO | 14 (6.4%) | 5 (6.8%) | **0 (0%)** |

> **Nota sobre el test set:** Los últimos 74 días (Oct 19 – Dic 31) tienen 0 días ALTO porque GDELT no tiene datos de Noviembre y Diciembre 2024 — esos días fueron imputados con cero, haciendo que el modelo los clasifique como calmos. Esta es una limitación conocida del dataset documentada en el EDA.

### Preprocesamiento aplicado

- **12 columnas eliminadas:** varianza cero (OpenSky), multicolinealidad r > 0.96, duplicadas
- **5 columnas con log1p:** `gdelt_n_eventos`, `gdelt_n_conflicto`, `ukmto_n_attacks`, `gdelt_n_conflicto_ma7`, `acled_region_fatalities` — reducción de skewness extrema (>3)
- **2 columnas binarizadas:** `acled_hormuz_n_violentos`, `maritime_n_hormuz` — más del 99% de ceros
- **StandardScaler:** ajustado exclusivamente sobre train, aplicado a val y test

---

## 2. Naive Bayes (GaussianNB)

### 2.1 Justificación del modelo

Naive Bayes es apropiado para este dataset porque:
- Funciona bien con features continuas (GaussianNB modela cada feature como distribución gaussiana por clase)
- Es robusto con pocos datos por clase — ALTO solo tiene 14 ejemplos en train
- Genera probabilidades calibradas que permiten cuantificar la incertidumbre del riesgo
- Supuesto de independencia condicional: aunque las features no son estrictamente independientes, la información proviene de 5 fuentes distintas (GDELT, UKMTO, Maritime, ACLED, temporal), lo que reduce la correlación estructural entre grupos

### 2.2 Tratamiento del desbalance

Con 14 ejemplos ALTO vs 134 MEDIO en train (ratio 9.6x), un prior uniforme haría que el modelo ignore ALTO casi por completo. Se usó **prior ajustado inversamente proporcional a la frecuencia**:

```python
priors = 1 / counts_por_clase
priors = priors / priors.sum()
# Resultado: BAJO=0.151  MEDIO=0.080  ALTO=0.768
```

El prior ALTO elevado (0.768) compensa el desbalance, forzando al modelo a considerar activamente esa clase.

### 2.3 Resultados

#### Validación (Ago 07 – Oct 18, 73 días)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BAJO | 0.23 | 1.00 | 0.38 | 13 |
| MEDIO | 0.90 | 0.16 | 0.28 | 55 |
| **ALTO** | **0.57** | **0.80** | **0.67** | **5** |

| Métrica global | Valor |
|---|---|
| **F1-Macro** | **0.440** |
| **F1-ALTO** | **0.667** |
| **Cohen's Kappa** | **0.146** |
| Accuracy | 0.356 |

#### Test (Oct 19 – Dic 31, 74 días)

| Métrica | Valor |
|---|---|
| F1-Macro | 0.666 |
| F1-ALTO | 0.000 (sin días ALTO en el período) |
| Cohen's Kappa | 0.358 |

### 2.4 Análisis de errores

El modelo comete **47 errores en validación**, todos con el mismo patrón:

```
MEDIO → predicho como BAJO
Condición: gdelt_goldstein_min = -10.0, ukmto_n_incidentes = 0, maritime_severidad_max = 0
```

**Diagnóstico:** Estos días tienen GDELT activo (goldstein_min = -10 = máximo conflicto) pero **sin incidentes marítimos directos**. El modelo Naive Bayes, al tratar cada feature como independiente, combina la señal GDELT elevada con la ausencia marítima y concluye BAJO. La realidad operacional es que el conflicto regional sin incidentes directos en Hormuz sigue siendo riesgo MEDIO.

También hay **2 errores MEDIO → ALTO** (Sep 15, Oct 10) donde el modelo sobre-estima el riesgo ante severidad marítima moderada (3–4) sin confirmación de otros indicadores.

**El único error crítico** es el día 2024-08-25 (ALTO real → predicho MEDIO): buque MV Sounion incendiado, severidad=4, sin datos UKMTO — Naive Bayes no logra integrar la señal marítima sola cuando las demás fuentes están en cero.

### 2.5 Fortalezas y debilidades

**Fortalezas:**
- Detecta el 80% de los días ALTO (recall=0.80) — operacionalmente valioso
- Genera probabilidades interpretables por clase
- Entrenamiento instantáneo, no requiere ajuste iterativo
- Robusto con n pequeño (14 ejemplos ALTO en train)

**Debilidades:**
- Supuesto de independencia viola la correlación real entre fuentes (r=0.43 GDELT–UKMTO)
- F1-Macro bajo (0.440) — sacrifica MEDIO para detectar ALTO
- El prior muy elevado en ALTO genera falsos positivos (precision ALTO = 0.57)
- No captura la interacción entre señales — la combinación GDELT alto + marítimo cero no es simplemente la suma de ambas

---

## 3. K-Means — Clustering Mensual

### 3.1 Justificación del modelo

K-Means se usa para **responder una pregunta distinta**: no predecir el riesgo de un día, sino **descubrir si 2024 tuvo regímenes temporales de conflicto diferenciados**. Cada punto en el espacio es un mes (12 puntos), representado por el promedio mensual de las 10 features más relevantes del conflicto.

Esto permite identificar:
- ¿Cuándo cambió el contexto geopolítico del Estrecho?
- ¿Qué meses son estructuralmente similares en términos de señales de riesgo?
- ¿Los eventos puntuales (ataque Irán→Israel en Abril) crearon un régimen sostenido o fue una anomalía?

### 3.2 Features usadas (10)

```
gdelt_n_eventos         gdelt_goldstein_mean    gdelt_goldstein_min
gdelt_n_conflicto       ukmto_n_incidentes      ukmto_n_attacks
maritime_n_incidentes   maritime_severidad_max  acled_region_n_explosions
acled_region_fatalities
```

### 3.3 Selección de k

| k | Inercia | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 68.32 | 0.347 | — |
| **3** | **44.67** | **0.401** | **0.776** |
| 4 | 29.32 | 0.358 | — |
| 5 | 22.25 | 0.332 | — |

**k=3** seleccionado por máximo Silhouette (0.401). Davies-Bouldin = 0.776 < 1.0 indica clusters bien separados.

### 3.4 Resultados: 3 regímenes de conflicto

#### Cluster 0 — Conflicto Activo (Abr, May, Jun, Jul, Ago, Sep, Oct)
**7 meses — 76.2% días MEDIO, 7.0% días ALTO**

El bloque central del año. Caracterizado por:
- GDELT `gdelt_n_eventos` media ≈ 4.7 (log1p), equivalente a ~110 eventos/día
- `gdelt_goldstein_min` = −10.0 en todos los meses (peor evento del día siempre crítico)
- `gdelt_goldstein_mean` entre −1.1 y −2.5 — ambiente mediático hostil sostenido
- UKMTO activo pero variable — Junio fue el mes más denso (14 incidentes)

**Eventos ancla:** Ataque Irán→Israel (Abril), respuesta Israel→Irán (Octubre), MV Sounion incendiado (Agosto). Estos eventos marcaron el inicio y sostuvieron la escalada durante 7 meses.

#### Cluster 2 — Pre-escalada (Ene, Feb, Mar)
**3 meses — 39.5% días MEDIO, 4.4% días ALTO**

Período de conflicto presente pero contenido:
- GDELT prácticamente ausente en Enero y Febrero (0.0 eventos/día) — conflicto aún no en agenda mediática internacional
- Marzo muestra los primeros signos con `gdelt_n_eventos` = 0.39 y Goldstein = −0.26
- Los días ALTO provienen principalmente de incidentes UKMTO/Maritime tempranos (Feb: MV Rubymar hundido)
- La ausencia de cobertura mediática no significa ausencia de riesgo — dato importante para el modelo

#### Cluster 1 — Artefacto de datos (Nov, Dic)
**2 meses — 0% días ALTO, 3.3% días MEDIO**

> **Advertencia:** Este cluster no representa un período de baja conflictividad real. Es el resultado de la **ausencia de datos GDELT** en Noviembre y Diciembre 2024. Al imputar con cero, el modelo los percibe como idénticos al Cluster 2 (pre-escalada) pero más extremos, formando su propio cluster.

Operacionalmente, no se puede afirmar que el Estrecho estuvo "en calma" en Nov–Dic sin datos GDELT.

### 3.5 Visualización del perfil

```
Feature                    Cluster 0    Cluster 2    Cluster 1
                           (Abr–Oct)    (Ene–Mar)    (Nov–Dic)
──────────────────────────────────────────────────────────────
gdelt_n_eventos              4.7           0.13          0.0    ← discriminador principal
gdelt_goldstein_min         -10.0          -0.12         0.0    ← señal de alarma
gdelt_n_conflicto            2.9           0.02          0.0
ukmto_n_incidentes           1.8           0.27          0.2
maritime_severidad_max       0.6           0.22          0.0
acled_region_fatalities     18.2          23.1           9.8    ← presión regional
```

### 3.6 Fortalezas y debilidades

**Fortalezas:**
- Descubre estructura temporal no supervisada — no necesita el target
- Confirma que Abril fue el punto de inflexión del año (inicio del Cluster 0)
- Silhouette aceptable (0.401) considerando que solo hay 12 puntos
- Interpretable: cada cluster tiene una narrativa geopolítica coherente

**Debilidades:**
- 12 puntos es muy poco para K-Means — resultados sensibles a outliers
- Cluster 1 (Nov–Dic) es un artefacto, no un hallazgo real
- K-Means asume clusters esféricos — la distribución real puede ser más compleja
- No captura variabilidad intra-mensual (meses con semanas de escalada y semanas calmas)

---

## 4. Comparación y conclusiones

| Aspecto | Naive Bayes | K-Means |
|---|---|---|
| **Tipo de tarea** | Clasificación supervisada | Clustering no supervisado |
| **Pregunta que responde** | ¿Qué nivel de riesgo tiene este día? | ¿Qué régimen de conflicto caracteriza este mes? |
| **Métrica principal** | F1-ALTO = 0.667 (val) | Silhouette = 0.401 |
| **Fortaleza** | Detecta 4 de 5 días ALTO | Identifica punto de inflexión en Abril |
| **Debilidad** | Falla en días con solo señal GDELT | Solo 12 puntos, Cluster Nov–Dic es artefacto |
| **Supuesto clave** | Independencia condicional de features | Clusters esféricos, varianza homogénea |

### Conclusión principal

Ambos modelos, con enfoques complementarios, confirman la misma hipótesis: **el Estrecho de Hormuz tuvo en 2024 un cambio de régimen de conflicto en Abril que se sostuvo hasta Octubre**, consistente con la escalada del conflicto Irán–Israel–Houthi.

Naive Bayes puede anticipar días de máximo riesgo (ALTO) con recall del 80%, utilizando principalmente la señal GDELT (`gdelt_goldstein_min`, `gdelt_n_eventos`) y la severidad marítima. K-Means revela que ese período de alto riesgo no fue una anomalía puntual sino un **régimen sostenido de 7 meses** estructuralmente diferente del primer trimestre del año.

La principal limitación compartida es la **ausencia de datos GDELT en Nov–Dic**, que hace imposible evaluar el comportamiento real de los modelos en el último bimestre.

---

*Modelos: `models/naive_bayes.pkl` | Figuras: `docs/model_figures/` (nb_confusion_val, nb_probs_val, kmeans_scatter, kmeans_elbow, kmeans_heatmap, kmeans_riesgo)*
