"""
EJEMPLO 3: Clasificación con KNN y Decision Trees
==================================================

Aprende a entrenar modelos de clasificación para predecir categorías.

OBJETIVO:
- Entender clasificación vs regresión
- Usar K-Nearest Neighbors (KNN)
- Usar Decision Trees
- Comparar algoritmos
- Evaluar con accuracy, confusion matrix, etc.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                            classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def crear_dataset_frutas():
    """Crea un dataset simple de frutas"""
    # Características: peso (g), diámetro (cm)
    # Clases: manzana, naranja, limón
    datos = {
        'peso_g': [150, 160, 145, 155, 165, 200, 210, 195, 205, 220, 80, 75, 85, 90, 70],
        'diametro_cm': [7.0, 7.5, 6.8, 7.2, 7.6, 8.5, 9.0, 8.2, 8.8, 9.2, 5.5, 5.2, 5.8, 6.0, 5.0],
        'fruta': ['manzana'] * 5 + ['naranja'] * 5 + ['limon'] * 5
    }
    return pd.DataFrame(datos)


def ejemplo_knn_basico():
    """Ejemplo básico de K-Nearest Neighbors"""
    print("=" * 70)
    print("EJEMPLO 1: K-Nearest Neighbors (KNN)")
    print("=" * 70)

    print("\nPROBLEMA: Clasificar frutas según peso y diámetro")
    print("\nALGORITMO KNN:")
    print("  1. Para predecir una nueva fruta")
    print("  2. Busca las K frutas más cercanas")
    print("  3. La fruta es de la clase más común entre las K vecinas")
    print("  Ejemplo: Si K=5 y hay 3 manzanas, 1 naranja, 1 limón → manzana")

    # Crear dataset
    print("\n1. PREPARAR DATOS")
    print("-" * 70)

    df = crear_dataset_frutas()
    print("\nDataset de frutas:")
    print(df.head(10))
    print(f"\nTotal: {len(df)} frutas")
    print(f"Clases: {df['fruta'].unique()}")

    # Separar X e y
    X = df[['peso_g', 'diametro_cm']]
    y = df['fruta']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain: {len(X_train)} frutas")
    print(f"Test:  {len(X_test)} frutas")

    # Entrenar KNN
    print("\n2. ENTRENAR MODELO KNN")
    print("-" * 70)

    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(X_train, y_train)

    print(f"Modelo KNN entrenado con K=3")
    print("  Busca las 3 frutas más cercanas para clasificar")

    # Predecir
    print("\n3. HACER PREDICCIONES")
    print("-" * 70)

    y_pred = modelo_knn.predict(X_test)

    print("\nPredicciones en test set:")
    for i, (peso, diam, real, pred) in enumerate(
        zip(X_test['peso_g'], X_test['diametro_cm'], y_test, y_pred)
    ):
        correcto = " " if real == pred else " "
        print(f"  {correcto} Fruta {i+1}: {peso}g, {diam}cm → "
              f"Predicho: {pred:8s}, Real: {real}")

    # Evaluar
    print("\n4. EVALUAR MODELO")
    print("-" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")

    # Matriz de confusión
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred, labels=['manzana', 'naranja', 'limon'])
    print("                  Predicho")
    print("              manzana naranja limon")
    print(f"Real manzana      {cm[0][0]}       {cm[0][1]}       {cm[0][2]}")
    print(f"     naranja      {cm[1][0]}       {cm[1][1]}       {cm[1][2]}")
    print(f"     limon        {cm[2][0]}       {cm[2][1]}       {cm[2][2]}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Predecir nueva fruta
    print("\n5. PREDECIR NUEVA FRUTA")
    print("-" * 70)

    nueva_fruta = [[170, 7.5]]  # 170g, 7.5cm
    prediccion = modelo_knn.predict(nueva_fruta)[0]

    print(f"\nFruta: peso={nueva_fruta[0][0]}g, diámetro={nueva_fruta[0][1]}cm")
    print(f"  Predicción: {prediccion}")

    return modelo_knn


def ejemplo_decision_tree():
    """Ejemplo de Árbol de Decisión"""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Decision Tree (Árbol de Decisión)")
    print("=" * 70)

    print("\nALGORITMO DECISION TREE:")
    print("  Serie de preguntas tipo 'Si-Entonces':")
    print("    ¿peso > 100g?")
    print("    ├─ SÍ: ¿diámetro > 8cm?")
    print("    │  ├─ SÍ: NARANJA")
    print("    │  └─ NO: MANZANA")
    print("    └─ NO: LIMÓN")

    # Preparar datos
    print("\n1. PREPARAR DATOS")
    print("-" * 70)

    df = crear_dataset_frutas()
    X = df[['peso_g', 'diametro_cm']]
    y = df['fruta']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar Decision Tree
    print("\n2. ENTRENAR DECISION TREE")
    print("-" * 70)

    modelo_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo_tree.fit(X_train, y_train)

    print("Modelo Decision Tree entrenado")
    print(f"  Profundidad máxima: 3")
    print(f"  Número de hojas: {modelo_tree.get_n_leaves()}")

    # Mostrar importancia de features
    print("\nImportancia de features:")
    for feature, importance in zip(X.columns, modelo_tree.feature_importances_):
        print(f"  {feature:15s}: {importance:.4f} ({importance*100:.1f}%)")

    # Predecir
    print("\n3. HACER PREDICCIONES")
    print("-" * 70)

    y_pred = modelo_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nPredicciones:")
    for peso, diam, real, pred in zip(
        X_test['peso_g'], X_test['diametro_cm'], y_test, y_pred
    ):
        correcto = "" if real == pred else ""
        print(f"  {correcto} {peso}g, {diam}cm → {pred:8s} (real: {real})")

    # Comparar con KNN
    print("\n4. COMPARAR CON KNN")
    print("-" * 70)

    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(X_train, y_train)
    y_pred_knn = modelo_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    print(f"\nResultados:")
    print(f"  Decision Tree: {accuracy:.2%}")
    print(f"  KNN (K=3):     {accuracy_knn:.2%}")

    return modelo_tree


def ejemplo_comparar_modelos():
    """Compara diferentes algoritmos"""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Comparar Múltiples Algoritmos")
    print("=" * 70)

    # Preparar datos
    df = crear_dataset_frutas()
    X = df[['peso_g', 'diametro_cm']]
    y = df['fruta']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definir modelos a probar
    modelos = {
        'KNN (K=1)': KNeighborsClassifier(n_neighbors=1),
        'KNN (K=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    }

    print("\nEntrenando y evaluando modelos...")
    print("-" * 70)

    resultados = {}

    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)

        # Evaluar
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'train': acc_train,
            'test': acc_test,
            'diff': abs(acc_train - acc_test)
        }

    # Mostrar resultados
    print("\nResultados:")
    print(f"{'Modelo':<20} {'Train':>8} {'Test':>8} {'Diff':>8} {'Diagnóstico'}")
    print("-" * 70)

    for nombre, res in resultados.items():
        diagnostico = ""
        if res['diff'] > 0.15:
            diagnostico = "Overfitting"
        elif res['test'] > 0.9:
            diagnostico = "Excelente"
        elif res['test'] > 0.7:
            diagnostico = "Bueno"
        else:
            diagnostico = "Regular"

        print(f"{nombre:<20} {res['train']:>7.1%} {res['test']:>7.1%} "
              f"{res['diff']:>7.1%}  {diagnostico}")

    # Mejor modelo
    print("\n ANÁLISIS:")
    mejor = max(resultados.items(), key=lambda x: x[1]['test'])
    print(f"  Mejor modelo en test: {mejor[0]} ({mejor[1]['test']:.1%})")

    overfitting = [n for n, r in resultados.items() if r['diff'] > 0.15]
    if overfitting:
        print(f"  Modelos con overfitting: {', '.join(overfitting)}")


def ejemplo_hiperparametros():
    """Muestra el impacto de hiperparámetros"""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Impacto de Hiperparámetros")
    print("=" * 70)

    df = crear_dataset_frutas()
    X = df[['peso_g', 'diametro_cm']]
    y = df['fruta']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Diferentes K en KNN
    print("\n1. KNN: Impacto de K (número de vecinos)")
    print("-" * 70)

    print(f"\n{'K':>3} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 30)

    for k in [1, 3, 5, 7, 9]:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)

        acc_train = modelo.score(X_train, y_train)
        acc_test = modelo.score(X_test, y_test)

        print(f"{k:>3} {acc_train:>10.1%} {acc_test:>10.1%}")

    print("\n Observación:")
    print("  - K muy pequeño (1): Alto overfitting, memoriza datos")
    print("  - K muy grande (9): Underfitting, demasiado simple")
    print("  - K=3-5: Balance ideal")

    # 2. Diferentes profundidades en Decision Tree
    print("\n2. Decision Tree: Impacto de max_depth")
    print("-" * 70)

    print(f"\n{'Depth':>5} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 30)

    for depth in [1, 2, 3, 5, 10]:
        modelo = DecisionTreeClassifier(max_depth=depth, random_state=42)
        modelo.fit(X_train, y_train)

        acc_train = modelo.score(X_train, y_train)
        acc_test = modelo.score(X_test, y_test)

        print(f"{depth:>5} {acc_train:>10.1%} {acc_test:>10.1%}")

    print("\n Observación:")
    print("  - depth=1: Muy simple, underfitting")
    print("  - depth=3-5: Buen balance")
    print("  - depth=10: Posible overfitting en datasets pequeños")


def ejemplo_metricas_detalladas():
    """Explica métricas de clasificación en detalle"""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Métricas de Clasificación Detalladas")
    print("=" * 70)

    df = crear_dataset_frutas()
    X = df[['peso_g', 'diametro_cm']]
    y = df['fruta']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # 1. Accuracy
    print("\n1. ACCURACY (Precisión General)")
    print("-" * 70)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy = {accuracy:.2%}")
    print(f"  Significa: {accuracy*100:.0f}% de predicciones fueron correctas")

    # 2. Confusion Matrix
    print("\n2. CONFUSION MATRIX (Matriz de Confusión)")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=['manzana', 'naranja', 'limon'])
    print("\n                    PREDICHO")
    print("              manzana  naranja  limon")
    print(f"REAL manzana     {cm[0][0]:2d}       {cm[0][1]:2d}       {cm[0][2]:2d}")
    print(f"     naranja     {cm[1][0]:2d}       {cm[1][1]:2d}       {cm[1][2]:2d}")
    print(f"     limon       {cm[2][0]:2d}       {cm[2][1]:2d}       {cm[2][2]:2d}")

    print("\n Interpretación:")
    print("  - Diagonal = predicciones correctas")
    print("  - Fuera diagonal = errores")
    print("  - Puedes ver qué clases confunde el modelo")

    # 3. Classification Report
    print("\n3. CLASSIFICATION REPORT")
    print("-" * 70)
    print("\n" + classification_report(y_test, y_pred))

    print(" Explicación detallada de métricas:")
    print("\n PRECISION (por clase):")
    print("   'De todas las veces que predije X, ¿cuántas eran realmente X?'")
    print("   Ejemplo: Si predije 'manzana' 10 veces y 8 eran manzanas → 80%")
    print("   Importante cuando: Los falsos positivos son malos")
    print("                      (Ej: clasificar email bueno como spam)")

    print("\n RECALL (por clase):")
    print("   'De todas las X reales, ¿cuántas detecté?'")
    print("   Ejemplo: Si hay 10 manzanas y detecté 9 → 90%")
    print("   Importante cuando: Los falsos negativos son malos")
    print("                      (Ej: no detectar un fraude)")

    print("\n F1-SCORE:")
    print("   Media armónica de precision y recall")
    print("   F1 = 2 × (precision × recall) / (precision + recall)")
    print("   Balance entre precision y recall")

    print("\n SUPPORT:")
    print("   Cuántas instancias reales hay de cada clase en test")

    # Ejemplo manual con la confusion matrix
    print("\n CÁLCULO MANUAL (ejemplo con 'manzana'):")
    # Total predicciones de manzana (columna 0)
    pred_manzana = cm[:, 0].sum()
    # Total reales de manzana (fila 0)
    real_manzana = cm[0, :].sum()
    # Correctos (diagonal)
    correctos_manzana = cm[0, 0]

    if pred_manzana > 0:
        precision_manzana = correctos_manzana / pred_manzana
        print(f"   Precision = {correctos_manzana}/{pred_manzana} = {precision_manzana:.2f}")

    if real_manzana > 0:
        recall_manzana = correctos_manzana / real_manzana
        print(f"   Recall    = {correctos_manzana}/{real_manzana} = {recall_manzana:.2f}")

    if pred_manzana > 0 and real_manzana > 0:
        f1_manzana = 2 * (precision_manzana * recall_manzana) / (precision_manzana + recall_manzana)
        print(f"   F1-score  = 2 × ({precision_manzana:.2f} × {recall_manzana:.2f}) / ({precision_manzana:.2f} + {recall_manzana:.2f}) = {f1_manzana:.2f}")


def errores_comunes():
    """Muestra errores comunes en clasificación"""
    print("\n" + "=" * 70)
    print("ERRORES COMUNES EN CLASIFICACIÓN")
    print("=" * 70)

    print("\n ERROR 1: No balancear clases")
    print("-" * 70)
    print("  Si tienes 90 manzanas y 10 limones:")
    print("  - Modelo que siempre predice 'manzana': 90% accuracy")
    print("  - Pero no aprendió nada sobre limones")
    print("  Solución: Usar stratify=y en train_test_split")

    print("\n ERROR 2: K muy alto o muy bajo en KNN")
    print("-" * 70)
    print("  - K=1: Overfitting, memoriza datos")
    print("  - K=N: Predice siempre la clase mayoritaria")
    print("  - Ideal: K=3-7 para empezar")

    print("\n ERROR 3: Decision Tree sin limitar profundidad")
    print("-" * 70)
    print("  DecisionTreeClassifier() sin max_depth")
    print("  → Crece hasta memorizar todos los datos")
    print("  → Overfitting garantizado")
    print("  Solución: Siempre usar max_depth=3-10")

    print("\n ERROR 4: No normalizar features en KNN")
    print("-" * 70)
    print("  KNN usa distancias. Si peso está en [50-200]")
    print("  y precio en [10000-50000], precio domina")
    print("  Solución: Normalizar con StandardScaler")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta todos los ejemplos"""
    print("\n" + "" * 35)
    print("CLASIFICACIÓN: KNN Y DECISION TREES")
    print("" * 35)

    # Ejecutar ejemplos
    ejemplo_knn_basico()
    ejemplo_decision_tree()
    ejemplo_comparar_modelos()
    ejemplo_hiperparametros()
    ejemplo_metricas_detalladas()
    errores_comunes()

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN: CLASIFICACIÓN")
    print("=" * 70)

    print("\n1. K-NEAREST NEIGHBORS (KNN)")
    print("  Código:")
    print("    from sklearn.neighbors import KNeighborsClassifier")
    print("    modelo = KNeighborsClassifier(n_neighbors=3)")
    print("  Ventajas: Simple, bueno con fronteras no lineales")
    print("  Desventajas: Lento, sensible a escala")

    print("\n2. DECISION TREE")
    print("  Código:")
    print("    from sklearn.tree import DecisionTreeClassifier")
    print("    modelo = DecisionTreeClassifier(max_depth=5)")
    print("  Ventajas: Interpretable, captura interacciones")
    print("  Desventajas: Tiende a overfitting")

    print("\n3. MÉTRICAS DE EVALUACIÓN")
    print("  - Accuracy: % predicciones correctas")
    print("  - Confusion Matrix: Ver dónde se equivoca")
    print("  - Precision/Recall: Métricas por clase")

    print("\n4. PARA PIEDRA, PAPEL O TIJERA:")
    print("  - 3 clases: piedra, papel, tijera")
    print("  - Features: frecuencias, lags, rachas, entropía")
    print("  - Probar KNN, Decision Tree, Random Forest")
    print("  - Objetivo: >50% accuracy (mejor que aleatorio)")

    print("\n" + "" * 35)


if __name__ == "__main__":
    main()
