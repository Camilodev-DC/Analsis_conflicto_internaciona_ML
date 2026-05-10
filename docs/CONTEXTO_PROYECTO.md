# Sistema de Inteligencia Multifuente — Contexto del Proyecto

**Curso:** Machine Learning 1 (ML1-2026I) · Pregrado Ciencia de Datos  
**Universidad:** Externado de Colombia · Docente: Julián Zuluaga  
**Fecha enunciado:** 14 de abril de 2026

---

## Objetivo central

Construir un **sistema de inteligencia multifuente** alrededor de la escalada del conflicto **Irán–Israel–EE.UU.**, usando exclusivamente fuentes gratuitas y públicas.

> **Pregunta orientadora:** ¿Hasta qué punto fuentes abiertas permiten detectar, clasificar o modelar episodios de escalada regional en el conflicto Irán–Israel–EE.UU.?

---

## Las 4 dimensiones que se evalúan

1. Conseguir y documentar datos reales
2. Construir un dataset integrado y utilizable
3. Formular y resolver una pregunta propia de ML
4. Comunicar resultados en un dashboard web desplegado

---

## Alcance mínimo obligatorio

- **3 a 5 fuentes** (mínimo: 1 textual + 1 estructurada/operativa + 1 contexto/movilidad/social)
- Pregunta propia formulada y justificada
- Unidad de análisis definida
- Mínimo **3 modelos** entrenados y comparados con métricas adecuadas
- **Dashboard desplegado por URL pública**

---

## Fuentes disponibles y su rol

| Fuente | Acceso | Tipo de datos | Rol en el proyecto |
|---|---|---|---|
| **ACLED** | API + OAuth2 | Eventos de conflicto estructurados | Labels / targets principales |
| UKMTO | Web scraping | Incidentes marítimos | Riesgo en Hormuz |
| **GDELT** | API (sin auth) | Noticias, tono, menciones geo | Corpus textual, NLP, embeddings |
| BBC RSS | RSS | Titulares | Corpus noticioso |
| Al Jazeera RSS | RSS | Titulares | Perspectiva regional |
| Google News RSS | RSS | Agregación | Diversidad de fuentes |
| **OpenSky** | API + OAuth2 | Vuelos (posición, altitud, vel) | Movilidad aérea |
| **AISStream** | WebSocket + API key | Embarcaciones AIS en tiempo real | Movilidad marítima |
| **NASA FIRMS** | API + MAP_KEY | Hotspots satelitales | Señales geoespaciales / térmicas |
| Cloudflare Radar | API | Anomalías de conectividad | Contexto digital |
| Bluesky | API | Posts públicos | Señal social / discurso |
| YouTube | API | Videos, comentarios | Contexto audiovisual |

---

## Esquema de normalización (schema mínimo del dataset)

Toda fuente debe terminar con estos campos para poder integrarse:

```
timestamp | source | country | lat | lon | text | event_type | value/score
```

---

## Tareas de ML posibles

### 1. Clasificación supervisada (tarea principal recomendada)
- Clasificar ventanas temporales por nivel de escalada
- Clasificar noticias/posts por tipo de narrativa
- Clasificar regiones por riesgo operativo

### 2. Regresión
- Score continuo de riesgo
- Intensidad mediática
- Predecir número de eventos en la siguiente ventana

### 3. Clustering (solo como apoyo exploratorio)
- Agrupar narrativas
- Identificar perfiles de días/regiones

---

## Modelos

### Obligatorios (baseline del curso)
- KNN, Naive Bayes, K-Means
- Linear / Logistic Regression
- Ridge, Lasso

### Adicionales permitidos
- Decision Tree, Random Forest, Gradient Boosting
- Embeddings (siempre comparados con baseline TF-IDF clásico)

---

## Flujo del proceso

```
FUENTES → DATASET → EDA → MODELOS → DASHBOARD
captura    integr.   explor.  compar.   despliegue
```

**Ruta metodológica sana:**
1. Escoger pregunta concreta
2. Validar 2-3 fuentes (muestra pequeña primero)
3. Construir tabla integrada con schema normalizado
4. EDA serio (distribuciones, calidad, sesgos)
5. Definir target y features
6. Entrenar y comparar modelos
7. Diseñar dashboard final

---

## Dashboard (requisitos mínimos)

- Descripción del problema
- Fuentes usadas
- Visualizaciones exploratorias
- Filtros: fecha, región, categoría, fuente
- Resultados del modelo + métricas
- Interpretación de hallazgos
- Limitaciones del sistema
- Herramientas: Streamlit, Dash, Next.js, Plotly, Folium, etc.

---

## Estructura del repositorio

```
data/
notebooks/ o src/
scripts/
models/
dashboard/
README.md
requirements.txt o pyproject.toml
```

---

## Criterios de evaluación

| Criterio | Preguntas clave |
|---|---|
| Formulación del problema | ¿Pregunta clara? ¿Tarea ML con sentido? ¿Unidad de análisis definida? |
| Calidad del dataset | ¿Fuentes justificadas? ¿Limpio? ¿Sesgos reconocidos? |
| Modelado y rigor | ¿Baseline? ¿Comparación con criterio? ¿Métricas correctas? ¿Análisis de errores? |
| Dashboard | ¿Aporta valor? ¿Conecta datos + análisis + modelo? |
| Calidad integral | ¿Reproducible? ¿Documentado? ¿Criterio y autonomía? |

---

## Errores frecuentes a evitar

- Bajar fuentes sin tener pregunta clara
- Construir el dashboard antes del dataset
- Usar modelos sin entender qué predicen
- Confundir visualización con machine learning
- No documentar limpieza y transformaciones
- No explicar de dónde salió el target
- No evaluar errores reales del modelo

---

## APIs ya investigadas

| API | Auth | Límite gratis | Lo que da |
|---|---|---|---|
| ACLED | OAuth2 Bearer | 5,000 filas/call | Eventos conflicto: actor, tipo, fecha, coords, fatalidades |
| GDELT | Sin auth | 250 artículos/call (raw sin límite) | Noticias + tono + geo, cada 15 min |
| OpenSky | OAuth2 Bearer | 400 créditos/día (anon) | Vuelos: posición, altitud, velocidad |
| NASA FIRMS | MAP_KEY en URL | 5,000 tx/10 min | Hotspots satelitales MODIS/VIIRS |
| AISStream | API key en WS | Gratis (beta) | Embarcaciones AIS en tiempo real (WebSocket) |
