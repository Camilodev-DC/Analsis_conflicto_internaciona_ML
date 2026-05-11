import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

def run_rf_robusto():
    """
    Entrena y evalúa el modelo Random Forest enfocado en máxima sensibilidad.
    Enfoque: Detección 100% de alertas críticas.
    """
    print("====================================================")
    print("  EJECUTANDO MODELO: RANDOM FOREST ROBUSTO")
    print("====================================================\n")

    # 1. Rutas y Carga
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)

    # 2. Feature Engineering (Incluyendo Sensor de Shock)
    df['indice_letalidad'] = df['acled_region_fatalities'] / (df['acled_region_n_eventos'] + 1)
    df['letalidad_ma7'] = df['indice_letalidad'].rolling(window=7).mean()
    df['shock_letalidad'] = df['indice_letalidad'] / (df['letalidad_ma7'] + 1e-6)
    
    # Rezagos
    cols_to_lag = ['acled_region_fatalities', 'gdelt_n_eventos', 'shock_letalidad']
    for col in cols_to_lag:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    risk_mapping = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2}
    df['target'] = df['nivel_riesgo'].map(risk_mapping)
    df['target_lag1'] = df['target'].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=['fecha', 'nivel_riesgo', 'target'])
    y = df['target']

    # 3. Split Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Balanceo SMOTE (Fuerte para evitar omisiones)
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Entrenamiento (Con pesos balanceados para el centinela)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_res, y_train_res)

    # 6. Reporte
    y_pred = rf.predict(X_test)
    print(">>> MÉTRICAS FINALES RANDOM FOREST ROBUSTO:")
    print(classification_report(y_test, y_pred, target_names=['BAJO', 'MEDIO', 'ALTO']))
    
    # Importancia de variables
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n>>> VARIABLES DETERMINANTES:")
    print(importances.head(3))

    return rf

if __name__ == "__main__":
    run_rf_robusto()
