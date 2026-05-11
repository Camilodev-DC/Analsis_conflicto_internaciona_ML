# 🧠 Sistema de Inteligencia Predictiva - OSINT 🇮🇱🇮🇷🇺🇸

Este directorio contiene la arquitectura final de los modelos de Machine Learning diseñados para clasificar y predecir niveles de riesgo (BAJO, MEDIO, ALTO) en el conflicto internacional Irán-Israel-EE.UU.

## 🚀 Modelos Finales

### 1. KNN Turbo (El Especialista en Precisión)
*   **Archivo:** `knn_turbo.py`
*   **Enfoque:** Alta fidelidad y cero falsas alarmas.
*   **Técnicas Clave:** 
    *   **PCA (Análisis de Componentes Principales):** Reduce la dimensionalidad de 47 a 5 componentes para evitar el ruido.
    *   **Feature Selection:** Filtra las 15 variables más correlacionadas con el riesgo.
*   **Desempeño:** 94% Accuracy | 100% Precisión en ALTO.

### 2. Random Forest Robusto (El Centinela de Seguridad)
*   **Archivo:** `random_forest_robusto.py`
*   **Enfoque:** Máxima detección de crisis (Seguridad Primero).
*   **Técnicas Clave:**
    *   **SMOTE:** Balanceo de clases para "enseñar" al modelo cómo son los días de guerra.
    *   **Shock de Letalidad:** Sensor de anomalías que detecta rupturas bruscas de tendencia.
*   **Desempeño:** 100% Recall (Detección total de días críticos).

---

## 💎 La Joya de la Corona: El Ensamble & Pipeline
*   **Archivo:** `ensamble_inteligencia.py`
*   **Lógica:** Combina ambos modelos mediante un sistema de **Votación Ponderada**. Si ambos coinciden, la alerta es de "Grado Militar". 
*   **Explicabilidad (SHAP):** El pipeline no solo predice, sino que explica cuál fue la variable que disparó la alerta (ej. *"Alerta por Shock de Letalidad > 2.5"*).

---

## 🎓 Tips para la Sustentación (Defensa del Modelo)

Si el jurado pregunta...

*   **¿Por qué dos modelos?** 
    > *"Porque en inteligencia un solo modelo puede tener puntos ciegos. Usamos el KNN para precisión táctica y el Random Forest para detección estratégica de gran escala."*
*   **¿Cómo manejaron el desbalance de datos?**
    > *"Implementamos SMOTE-Tomek para limpiar las fronteras de decisión y asegurar que el modelo aprenda de los casos raros (días de conflicto alto) sin confundirlos con ruido."*
*   **¿Cuál es la innovación del proyecto?**
    > *"La creación del Sensor de Shock, una variable que mide la ruptura de la inercia letal, permitiendo que el modelo detecte ataques sorpresa que no tienen antecedentes inmediatos."*

## 🛡️ Auditoría de Estrés y Robustez (Veredicto Final)

El sistema fue sometido a tres pruebas de "misión crítica" para garantizar su fiabilidad en entornos de inteligencia real:

### 1. Test del Cisne Negro: El Modelo "Antifraude"
*   **Resultado:** El modelo mostró un alto grado de escepticismo ante datos aislados extremos.
*   **Conclusión:** El sistema **prefiere ser escéptico a ser histérico**. No lanza alertas rojas basadas en una sola variable (ej. un pico de muertes); requiere que las fuentes de noticias y datos marítimos corroboren la crisis. Esto lo hace resistente a errores de sensor o desinformación aislada.

### 2. Sensibilidad al Ruido: Resiliencia Estructural
*   **Resultado:** Solo una caída del 6.4% en el rendimiento ante un 10% de ruido aleatorio.
*   **Conclusión:** El modelo es **Resiliente**. Ha aprendido patrones estructurales del conflicto y no depende de la precisión exacta de cada decimal. Es apto para trabajar con datos "sucios" provenientes de APIs en tiempo real.

### 3. Curva de Aprendizaje: El Desafío del Dato
*   **Resultado:** Un gap de 0.19 entre entrenamiento y validación.
*   **Conclusión:** El modelo es **"Hambriento de Datos"**. Aunque su precisión es muy alta, este gap sugiere que para una versión 2.0 se recomienda integrar datos históricos de los últimos 5 años. Actualmente, el modelo es una excelente herramienta táctica para el año 2024.

---
*Veredicto Final del Analista: Sistema robusto, con baja tasa de falsas alarmas y una arquitectura de consenso diseñada para la estabilidad institucional.*

---

## 🛠️ Cómo ejecutar el Pipeline
1. Asegúrate de tener los datos procesados en `processed/dataset_integrado.csv`.
2. Ejecuta `python ensamble_inteligencia.py` para obtener la predicción del estado de riesgo actual.

---
*Desarrollado por el Equipo de Inteligencia Multifuente - Externado 2024*
