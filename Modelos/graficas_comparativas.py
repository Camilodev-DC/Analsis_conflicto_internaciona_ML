import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from knn_turbo import run_knn_turbo
from random_forest_robusto import run_rf_robusto

def generar_graficas_comparativas():
    print("Generando Graficas Comparativas de Alto Nivel...")
    
    # 1. Obtener modelos y datos
    # Re-ejecutamos brevemente para obtener los resultados frescos
    knn, pca, scaler, selector = run_knn_turbo()
    rf = run_rf_robusto()
    
    # Carga de datos para validación
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "processed", "dataset_integrado.csv")
    df = pd.read_csv(data_path)
    
    # Preprocesamiento rápido para obtener X_test e y_test
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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Predicciones
    # Para KNN
    X_test_sel = selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_sel)
    X_test_pca = pca.transform(X_test_scaled)
    y_pred_knn = knn.predict(X_test_pca)
    
    # Para RF
    y_pred_rf = rf.predict(X_test)

    # --- GRÁFICA 1: MATRICES DE CONFUSIÓN LADO A LADO ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=ax1,
                xticklabels=['BAJO', 'MEDIO', 'ALTO'], yticklabels=['BAJO', 'MEDIO', 'ALTO'])
    ax1.set_title('Matriz de Confusión: KNN TURBO\n(Enfoque Precisión)')
    ax1.set_xlabel('Predicho')
    ax1.set_ylabel('Real')

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['BAJO', 'MEDIO', 'ALTO'], yticklabels=['BAJO', 'MEDIO', 'ALTO'])
    ax2.set_title('Matriz de Confusión: RF ROBUSTO\n(Enfoque Seguridad/Recall)')
    ax2.set_xlabel('Predicho')
    ax2.set_ylabel('Real')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'comparativa_matrices.png'))
    print("Guardada: comparativa_matrices.png")

    # --- GRAFICA 2: IMPORTANCIA DE VARIABLES VS PCA LOADINGS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RF Importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
    importances.plot(kind='barh', color='darkorange', ax=ax1)
    ax1.set_title('Random Forest: Top 10 Variables')
    ax1.set_xlabel('Importancia Relativa')

    # PCA Loadings (Componente 1)
    # Obtenemos los nombres de las variables que pasaron el selector
    selected_features = X.columns[selector.get_support()]
    loadings = pd.Series(pca.components_[0], index=selected_features).abs().sort_values(ascending=True)
    loadings.plot(kind='barh', color='darkgreen', ax=ax2)
    ax2.set_title('PCA (KNN): Peso de Variables en CP1')
    ax2.set_xlabel('Valor Absoluto del Loading')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'importancia_vs_pca.png'))
    print("Guardada: importancia_vs_pca.png")

if __name__ == "__main__":
    generar_graficas_comparativas()
