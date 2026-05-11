import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def train_knn_turbo():
    print("====================================================")
    print("  Experimento 5: KNN TURBO (PCA + Feature Selection)")
    print("====================================================\n")

    # 1. Cargar datos y crear las variables que ya sabemos que funcionan
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

    # 2. SELECCIÓN DE VARIABLES (Top 15)
    # Filtramos el ruido antes de entrar al KNN
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    print(f"2. Selección de Variables: Reducido de {X.shape[1]} a 15 variables clave.")

    # 3. Split Estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. ESCALADO (Vital para PCA y KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. PCA (Reducción de Dimensionalidad a 5 componentes)
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"3. PCA aplicado: Comprimido a 5 Componentes Principales.")

    # 6. BALANCEO (SMOTE)
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train)

    # 7. ENTRENAMIENTO KNN OPTIMIZADO
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train_res, y_train_res)

    # 8. Evaluación
    y_pred = knn.predict(X_test_pca)
    
    print("\n>>> RESULTADOS KNN TURBO (CON PCA Y SELECCIÓN):")
    print(classification_report(y_test, y_pred, target_names=['BAJO', 'MEDIO', 'ALTO'], zero_division=0))
    
    print("\n>>> MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_knn_turbo()
