import pandas as pd
import numpy as np
import os
import joblib
from knn_turbo import run_knn_turbo
from random_forest_robusto import run_rf_robusto

class SistemaEnsambleOSINT:
    """
    Sistema maestro que combina KNN y Random Forest para una decisión por consenso.
    Incluye lógica de explicabilidad y pipeline de predicción.
    """
    def __init__(self):
        print("Inicializando Sistema de Ensamble de Inteligencia...")
        self.knn_model, self.pca, self.scaler, self.selector = run_knn_turbo()
        self.rf_model = run_rf_robusto()
        self.risk_labels = {0: 'BAJO', 1: 'MEDIO', 2: 'ALTO'}

    def predecir_con_consenso(self, datos_nuevos):
        """
        Ejecuta la predicción en ambos modelos y decide por consenso.
        """
        # Preparación para KNN (Selección -> Escalado -> PCA)
        datos_selected = self.selector.transform(datos_nuevos)
        datos_scaled = self.scaler.transform(datos_selected)
        datos_pca = self.pca.transform(datos_scaled)
        
        pred_knn = self.knn_model.predict(datos_pca)[0]
        
        # Preparación para RF (Directo)
        pred_rf = self.rf_model.predict(datos_nuevos)[0]
        
        # LÓGICA DE CONSENSO
        if pred_knn == pred_rf:
            veredicto = self.risk_labels[pred_knn]
            confianza = "ALTA (Consenso Total)"
        else:
            # Si hay desacuerdo, priorizamos la seguridad del Random Forest en riesgos altos
            # o el promedio ponderado. Aquí usamos el RF como "Veto" si es ALTO.
            final_risk = max(pred_knn, pred_rf)
            veredicto = self.risk_labels[final_risk]
            confianza = f"MEDIA (Discrepancia entre modelos: KNN={self.risk_labels[pred_knn]}, RF={self.risk_labels[pred_rf]})"
            
        return veredicto, confianza

    def explicar_alerta(self, fila_datos):
        """
        Analiza las variables clave para dar una explicación humana de la alerta.
        """
        explicacion = []
        if fila_datos['shock_letalidad'].values[0] > 1.5:
            explicacion.append(f"⚠️ SHOCK DETECTADO: La letalidad hoy es {fila_datos['shock_letalidad'].values[0]:.2f} veces mayor al promedio semanal.")
        
        if fila_datos['gdelt_n_eventos'].values[0] > 200:
            explicacion.append(f"📢 RUIDO MEDIÁTICO: Volumen inusual de noticias detectado ({fila_datos['gdelt_n_eventos'].values[0]} eventos).")
            
        if fila_datos['target_lag1'].values[0] == 2:
            explicacion.append("⏳ INERCIA CRÍTICA: El día de ayer ya se encontraba en riesgo ALTO.")
            
        return " | ".join(explicacion) if explicacion else "Situación dentro de los parámetros normales."

def ejecutar_pipeline_ejemplo():
    """
    Simula la predicción para el día de 'MAÑANA' usando los últimos datos conocidos.
    """
    print("\n" + "="*50)
    print("PIPELINE DE PREDICCION EN TIEMPO REAL")
    print("="*50)
    
    sistema = SistemaEnsambleOSINT()
    
    # Tomamos el último día del dataset como ejemplo de 'Hoy' para predecir 'Mañana'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)
    
    # (Simulación de preprocesamiento de la última fila)
    # En un caso real, aquí cargaríamos los datos de la API de hoy
    ultima_fila = df.tail(1).copy()
    # Ajustamos variables para el ejemplo
    ultima_fila['indice_letalidad'] = 0.5
    ultima_fila['letalidad_ma7'] = 0.2
    ultima_fila['shock_letalidad'] = 2.5 # Forzamos un Shock para ver la explicación
    ultima_fila['target_lag1'] = 1
    
    X_input = ultima_fila.drop(columns=['fecha', 'nivel_riesgo'])
    if 'target' in X_input.columns: X_input = X_input.drop(columns=['target'])

    veredicto, confianza = sistema.predecir_con_consenso(X_input)
    explicacion = sistema.explicar_alerta(ultima_fila)
    
    print(f"\nPREDICCION PARA MANANA: {veredicto}")
    print(f"CONFIANZA: {confianza}")
    print(f"ANALISIS: {explicacion}")
    print("\n" + "="*50)

if __name__ == "__main__":
    ejecutar_pipeline_ejemplo()
