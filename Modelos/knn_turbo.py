import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import os

def run_knn_turbo():
    """
    Entrena y evalúa el modelo KNN optimizado con PCA y Feature Selection.
    Enfoque: Alta precisión y reducción de la dimensionalidad.
    """
    print("====================================================")
    print("  EJECUTANDO MODELO: KNN TURBO (OSINT)")
    print("====================================================\n")

    # 1. Rutas y Carga
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)

    # 2. Feature Engineering
    df['indice_letalidad'] = df['acled_region_fatalities'] / (df['acled_region_n_eventos'] + 1)
    df['letalidad_ma7'] = df['indice_letalidad'].rolling(window=7).mean()
    df['shock_letalidad'] = df['indice_letalidad'] / (df['letalidad_ma7'] + 1e-6)
    
    # Lags clave
    cols_to_lag = ['acled_region_fatalities', 'gdelt_n_eventos', 'shock_letalidad']
    for col in cols_to_lag:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    # Target Encoding
    risk_mapping = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2}
    df['target'] = df['nivel_riesgo'].map(risk_mapping)
    df['target_lag1'] = df['target'].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=['fecha', 'nivel_riesgo', 'target'])
    y = df['target']

    # 3. Selección de Variables (Reducción de ruido)
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)

    # 4. Split Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Escalado y PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 6. Balanceo SMOTE
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train)

    # 7. Entrenamiento
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train_res, y_train_res)

    # 8. Reporte
    y_pred = knn.predict(X_test_pca)
    print(">>> MÉTRICAS FINALES KNN TURBO:")
    print(classification_report(y_test, y_pred, target_names=['BAJO', 'MEDIO', 'ALTO']))
    
    return knn, pca, scaler, selector

if __name__ == "__main__":
    run_knn_turbo()
