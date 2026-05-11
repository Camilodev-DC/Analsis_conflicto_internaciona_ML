import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif

def train_rf_final():
    print("====================================================")
    print("  EXPERIMENTO FINAL: Random Forest + SMOTE-Tomek")
    print("====================================================\n")

    # 1. Cargar datos y crear variables
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "..", "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)

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

    # 2. SELECCIÓN DE VARIABLES (Top 15 para reducir ruido)
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    print(f"2. Variables seleccionadas (Top 15): {list(selected_cols)}")

    # 3. Split Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. BALANCEO HÍBRIDO (SMOTE-Tomek)
    # Crea ejemplos y luego borra los que causan confusión en la frontera
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
    print(f"3. SMOTE-Tomek aplicado: Entrenamiento balanceado y limpio.")

    # 5. RANDOM FOREST FINAL (Con Pesos Calibrados)
    # Le damos un poco más de peso a la clase ALTO pero cuidando la precisión
    custom_weights = {0: 1, 1: 1.5, 2: 3} 
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight=custom_weights,
        random_state=42
    )
    rf.fit(X_train_res, y_train_res)

    # 6. Evaluación
    y_pred = rf.predict(X_test)
    
    print("\n>>> RESULTADOS RANDOM FOREST FINAL (SMOTE-TOMEK):")
    print(classification_report(y_test, y_pred, target_names=['BAJO', 'MEDIO', 'ALTO'], zero_division=0))
    
    print("\n>>> MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, y_pred))

    # Análisis del 28 de enero
    df_test = df.iloc[y_test.index]
    success_jan28 = not df_test[(df_test['fecha'] == '2024-01-28') & (y_pred == 2)].empty
    if success_jan28:
        print("\nPERFORMANCE: El modelo mantuvo la detección del 28 de enero.")
    else:
        print("\nPERFORMANCE: El modelo perdió el 28 de enero (revisar pesos).")

if __name__ == "__main__":
    train_rf_final()
