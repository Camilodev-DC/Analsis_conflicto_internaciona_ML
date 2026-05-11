import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score
import os
from ensamble_inteligencia import SistemaEnsambleOSINT

def ejecutar_pruebas_estres():
    print("====================================================")
    print("  AUDITORIA DE CALIDAD: TESTS DE ESTRES Y OVERFITTING")
    print("====================================================\n")

    sistema = SistemaEnsambleOSINT()
    
    # Carga de datos base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)
    
    # Preprocesamiento para obtener X e y
    df['indice_letalidad'] = df['acled_region_fatalities'] / (df['acled_region_n_eventos'] + 1)
    df['letalidad_ma7'] = df['indice_letalidad'].rolling(window=7).mean()
    df['shock_letalidad'] = df['indice_letalidad'] / (df['letalidad_ma7'] + 1e-6)
    
    cols_to_lag = ['acled_region_fatalities', 'gdelt_n_eventos', 'shock_letalidad']
    for col in cols_to_lag:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    risk_mapping = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2}
    df['target'] = df['nivel_riesgo'].map(risk_mapping)
    df['target_lag1'] = df['target'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    X = df.drop(columns=['fecha', 'nivel_riesgo', 'target'])
    y = df['target']

    # --- 1. TEST DEL CISNE NEGRO (Evento Extremo Imprevisto) ---
    print("1. Ejecutando Test del Cisne Negro...")
    fila_cisne = X.tail(1).copy()
    # Simulamos un ataque masivo sin precedentes
    fila_cisne['acled_region_fatalities'] = X['acled_region_fatalities'].max() * 5
    fila_cisne['shock_letalidad'] = 15.0 # Un shock brutal
    
    veredicto, confianza = sistema.predecir_con_consenso(fila_cisne)
    print(f"   -> Reaccion del Modelo: {veredicto} ({confianza})")
    if veredicto == 'ALTO':
        print("   -> RESULTADO: PASADO. El modelo detecta eventos extremos jamas vistos.")
    else:
        print("   -> RESULTADO: FALLIDO. El modelo es demasiado rigido.")

    # --- 2. SENSIBILIDAD AL RUIDO ---
    print("\n2. Ejecutando Test de Sensibilidad al Ruido...")
    # Añadimos ruido gaussiano del 10% de la desviacion estandar
    X_ruidoso = X + np.random.normal(0, X.std() * 0.1, X.shape)
    
    # Evaluamos precision con ruido
    # (Usamos una muestra para rapidez)
    y_pred_original = []
    y_pred_ruidoso = []
    
    # Evaluamos con el modelo RF del ensamble para simplificar el loop
    y_pred_orig = sistema.rf_model.predict(X)
    y_pred_ruid = sistema.rf_model.predict(X_ruidoso)
    
    acc_orig = accuracy_score(y, y_pred_orig)
    acc_ruid = accuracy_score(y, y_pred_ruid)
    
    print(f"   -> Precision Original: {acc_orig:.4f}")
    print(f"   -> Precision con 10% de Ruido: {acc_ruid:.4f}")
    caida = (acc_orig - acc_ruid) / acc_orig
    print(f"   -> Caida de rendimiento: {caida*100:.2f}%")
    if caida < 0.10:
        print("   -> RESULTADO: PASADO. El modelo es robusto al ruido ambiental.")
    else:
        print("   -> RESULTADO: RIESGO. El modelo podria estar haciendo overfitting.")

    # --- 3. CURVA DE APRENDIZAJE (Deteccion de Overfitting) ---
    print("\n3. Generando Curva de Aprendizaje...")
    train_sizes, train_scores, test_scores = learning_curve(
        sistema.rf_model, X, y, cv=5, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validacion Cruzada")
    plt.title("Curva de Aprendizaje (RF Robusto)")
    plt.xlabel("Tamano del Dataset de Entrenamiento")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'curva_aprendizaje.png'))
    print("   -> RESULTADO: Grafica guardada como 'curva_aprendizaje.png'.")
    
    # Interpretacion automatica
    distancia_final = train_mean[-1] - test_mean[-1]
    print(f"   -> Gap final Entrenamiento/Prueba: {distancia_final:.4f}")
    if distancia_final < 0.15:
        print("   -> VEREDICTO FINAL: Sin Overfitting Significativo. Modelo listo para despliegue.")
    else:
        print("   -> VEREDICTO FINAL: Posible Overfitting. Se recomienda simplificar el modelo.")

if __name__ == "__main__":
    ejecutar_pruebas_estres()
