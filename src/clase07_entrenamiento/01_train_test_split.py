"""
EJEMPLO 1: Train/Test Split
============================

Aprende a dividir tus datos correctamente para entrenamiento y evaluación.

OBJETIVO:
- Entender por qué necesitamos dividir los datos
- Usar train_test_split de sklearn
- Ver el impacto de random_state
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def ejemplo_basico():
    """Ejemplo básico de división de datos"""
    print("=" * 70)
    print("EJEMPLO 1: División Básica de Datos")
    print("=" * 70)

    # Crear un dataset simple
    print("\n1. CREAR DATASET")
    print("-" * 70)

    # Supongamos que tenemos datos de estudiantes
    datos = {
        'horas_estudio': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'horas_sueno': [8, 7, 7, 6, 6, 5, 5, 4, 4, 3],
        'nota_examen': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    }

    df = pd.DataFrame(datos)
    print("\nDataset completo:")
    print(df)
    print(f"\nTotal de muestras: {len(df)}")

    # Separar features (X) y target (y)
    print("\n2. SEPARAR FEATURES Y TARGET")
    print("-" * 70)

    X = df[['horas_estudio', 'horas_sueno']]  # Features (entrada)
    y = df['nota_examen']                      # Target (lo que queremos predecir)

    print("\nFeatures (X):")
    print(X.head())
    print(f"Shape: {X.shape}")  # (10 filas, 2 columnas)

    print("\nTarget (y):")
    print(y.head())
    print(f"Shape: {y.shape}")  # (10 valores,)

    # Dividir en train y test
    print("\n3. DIVIDIR EN TRAIN Y TEST")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% para test, 80% para train
        random_state=42     # Semilla para reproducibilidad
    )

    print(f"\nDatos de ENTRENAMIENTO:")
    print(f"  X_train: {X_train.shape} = {len(X_train)} muestras")
    print(f"  y_train: {y_train.shape} = {len(y_train)} valores")

    print(f"\nDatos de TEST:")
    print(f"  X_test:  {X_test.shape} = {len(X_test)} muestras")
    print(f"  y_test:  {y_test.shape} = {len(y_test)} valores")

    print("\nX_train (datos para entrenar):")
    print(X_train)

    print("\nX_test (datos para evaluar):")
    print(X_test)


def ejemplo_random_state():
    """Demuestra la importancia de random_state"""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Importancia de random_state")
    print("=" * 70)

    # Dataset pequeño
    X = np.array([[i] for i in range(10)])  # [0, 1, 2, ..., 9]
    y = np.array([i * 2 for i in range(10)])  # [0, 2, 4, ..., 18]

    print("\nDatos originales:")
    print(f"X = {X.flatten()}")
    print(f"y = {y}")

    # Sin random_state (resultado diferente cada vez)
    print("\n1. SIN random_state (resultados aleatorios):")
    print("-" * 70)

    for i in range(3):
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3)
        print(f"Ejecución {i+1} - Test set: {X_test.flatten()}")

    # Con random_state (resultado reproducible)
    print("\n2. CON random_state=42 (resultados consistentes):")
    print("-" * 70)

    for i in range(3):
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"Ejecución {i+1} - Test set: {X_test.flatten()}")

    print("\n CONCLUSIÓN:")
    print("  - Sin random_state: resultados diferentes cada vez")
    print("  - Con random_state: resultados reproducibles")
    print("  - Útil para depurar y comparar experimentos")


def ejemplo_estratificado():
    """Ejemplo de stratified split para datasets desbalanceados"""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Stratified Split (División Estratificada)")
    print("=" * 70)

    # Dataset desbalanceado (más de clase A que de B)
    X = np.arange(100).reshape(-1, 1)
    y = ['A'] * 80 + ['B'] * 20  # 80% clase A, 20% clase B

    print("\nDataset DESBALANCEADO:")
    print(f"  Total: {len(y)} muestras")
    print(f"  Clase A: {y.count('A')} muestras (80%)")
    print(f"  Clase B: {y.count('B')} muestras (20%)")

    # División normal
    print("\n1. División NORMAL (no estratificada):")
    print("-" * 70)

    _, _, _, y_test_normal = train_test_split(X, y, test_size=0.2, random_state=42)

    count_a = sum(1 for label in y_test_normal if label == 'A')
    count_b = sum(1 for label in y_test_normal if label == 'B')

    print(f"Test set:")
    print(f"  Clase A: {count_a} / {len(y_test_normal)} = {count_a/len(y_test_normal):.1%}")
    print(f"  Clase B: {count_b} / {len(y_test_normal)} = {count_b/len(y_test_normal):.1%}")
    print("  Distribución puede variar del original")

    # División estratificada
    print("\n2. División ESTRATIFICADA:")
    print("-" * 70)

    _, _, _, y_test_strat = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,      # ¡Mantiene la proporción!
        random_state=42
    )

    count_a = sum(1 for label in y_test_strat if label == 'A')
    count_b = sum(1 for label in y_test_strat if label == 'B')

    print(f"Test set:")
    print(f"  Clase A: {count_a} / {len(y_test_strat)} = {count_a/len(y_test_strat):.1%}")
    print(f"  Clase B: {count_b} / {len(y_test_strat)} = {count_b/len(y_test_strat):.1%}")
    print("  Mantiene la proporción 80/20 del dataset original")

    print("\n CUÁNDO USAR stratify:")
    print("  - Cuando tienes clases desbalanceadas")
    print("  - Asegura que test set sea representativo")
    print("  - Especialmente importante con pocas muestras")


def ejemplo_multiples_splits():
    """Ejemplo de diferentes proporciones de split"""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Diferentes Proporciones de Split")
    print("=" * 70)

    # Dataset
    X = np.arange(1000).reshape(-1, 1)
    y = np.arange(1000)

    print("\nDataset: 1000 muestras")
    print("\nProbando diferentes proporciones:")
    print("-" * 70)

    proporciones = [0.1, 0.2, 0.3, 0.4]

    for prop in proporciones:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=prop,
            random_state=42
        )

        print(f"\ntest_size = {prop:.1%}")
        print(f"  Train: {len(X_train):4d} muestras ({len(X_train)/1000:.1%})")
        print(f"  Test:  {len(X_test):4d} muestras ({len(X_test)/1000:.1%})")

    print("\n RECOMENDACIONES:")
    print("  - Dataset grande (>10k):    test_size = 0.1 (10%)")
    print("  - Dataset medio (1k-10k):   test_size = 0.2 (20%)")
    print("  - Dataset pequeño (<1k):    test_size = 0.2-0.3 (20-30%)")
    print("  - Muy pocos datos (<100):   Usar validación cruzada")


def ejemplo_tres_conjuntos():
    """División en train, validation y test"""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Train, Validation y Test Sets")
    print("=" * 70)

    X = np.arange(1000).reshape(-1, 1)
    y = np.arange(1000)

    print("\nDataset: 1000 muestras")
    print("\nObjetivo: Dividir en 3 conjuntos")
    print("  - Train:      70% (700 muestras)")
    print("  - Validation: 15% (150 muestras)")
    print("  - Test:       15% (150 muestras)")

    # Primera división: separar test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42
    )

    # Segunda división: separar train y validation
    # De los 850 restantes, validation será ~17.6% para obtener 15% del total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,  # 150/850 ≈ 0.176
        random_state=42
    )

    print("\nResultado:")
    print(f"  Train:      {len(X_train):4d} muestras ({len(X_train)/1000:.1%})")
    print(f"  Validation: {len(X_val):4d} muestras ({len(X_val)/1000:.1%})")
    print(f"  Test:       {len(X_test):4d} muestras ({len(X_test)/1000:.1%})")

    print("\n USO DE CADA CONJUNTO:")
    print("  - Train:      Entrenar el modelo")
    print("  - Validation: Ajustar hiperparámetros")
    print("  - Test:       Evaluación final (solo al final)")


def errores_comunes():
    """Muestra errores comunes al hacer split"""
    print("\n" + "=" * 70)
    print("ERRORES COMUNES Y CÓMO EVITARLOS")
    print("=" * 70)

    print("\n ERROR 1: Entrenar y evaluar con los mismos datos")
    print("-" * 70)
    print("  MAL:")
    print("    modelo.fit(X, y)")
    print("    accuracy = modelo.score(X, y)  # ¡Usa los mismos datos!")
    print("")
    print("  BIEN:")
    print("    X_train, X_test, y_train, y_test = train_test_split(X, y)")
    print("    modelo.fit(X_train, y_train)")
    print("    accuracy = modelo.score(X_test, y_test)  # Datos nuevos")

    print("\n ERROR 2: Olvidar random_state")
    print("-" * 70)
    print("  Problema: Resultados no reproducibles")
    print("  Solución: Siempre usar random_state=42 (o cualquier número)")

    print("\n ERROR 3: No estratificar con clases desbalanceadas")
    print("-" * 70)
    print("  Problema: Test set no representativo")
    print("  Solución: Usar stratify=y cuando hay desbalance")

    print("\n ERROR 4: Test set muy pequeño o muy grande")
    print("-" * 70)
    print("  Muy pequeño (<5%):   Evaluación poco confiable")
    print("  Muy grande (>30%):   Menos datos para entrenar")
    print("  Ideal: 20% para test")

    print("\n ERROR 5: Usar test set para tomar decisiones")
    print("-" * 70)
    print("  MAL: Probar muchos modelos en test, elegir el mejor")
    print("  BIEN: Usar validation set para elegir, test solo al final")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta todos los ejemplos"""
    print("\n" + "" * 35)
    print("TRAIN/TEST SPLIT: CONCEPTOS Y EJEMPLOS")
    print("" * 35)

    # Ejecutar ejemplos
    ejemplo_basico()
    ejemplo_random_state()
    ejemplo_estratificado()
    ejemplo_multiples_splits()
    ejemplo_tres_conjuntos()
    errores_comunes()

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print("\nConceptos clave:")
    print("  1. Siempre dividir datos en train y test")
    print("  2. Test set NUNCA se usa durante entrenamiento")
    print("  3. Usar random_state para reproducibilidad")
    print("  4. Estratificar cuando hay clases desbalanceadas")
    print("  5. Proporción típica: 80% train, 20% test")

    print("\nCódigo básico:")
    print("  from sklearn.model_selection import train_test_split")
    print("  X_train, X_test, y_train, y_test = train_test_split(")
    print("      X, y, test_size=0.2, random_state=42")
    print("  )")

    print("\n" + "" * 35)


if __name__ == "__main__":
    main()
