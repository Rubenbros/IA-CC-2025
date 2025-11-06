"""
Ejemplo de cómo usar la clase PPTFeatureEngineering con un CSV real

Este script muestra el workflow completo desde cargar un CSV
hasta generar todas las features para entrenar un modelo.
"""

import pandas as pd
from feature_engineering_ppt import PPTFeatureEngineering

def ejemplo_completo_desde_csv():
    """Ejemplo completo: CSV → Features → DataFrame listo para ML"""

    print("=" * 70)
    print("EJEMPLO: De CSV con 3 columnas a DataFrame con 30+ features")
    print("=" * 70)

    # ========== PASO 1: Crear CSV de ejemplo ==========
    print("\n[PASO 1] Creando CSV de ejemplo...")

    # Simular datos que los alumnos tendrían
    datos_ejemplo = {
        'numero_ronda': list(range(1, 21)),
        'jugada_jugador': [
            'piedra', 'papel', 'piedra', 'tijera', 'papel',
            'piedra', 'papel', 'papel', 'tijera', 'piedra',
            'papel', 'tijera', 'piedra', 'papel', 'tijera',
            'piedra', 'papel', 'tijera', 'piedra', 'papel'
        ],
        'jugada_oponente': [
            'papel', 'piedra', 'tijera', 'tijera', 'piedra',
            'papel', 'papel', 'tijera', 'piedra', 'papel',
            'tijera', 'papel', 'piedra', 'tijera', 'papel',
            'tijera', 'piedra', 'tijera', 'papel', 'tijera'
        ]
    }

    df_original = pd.DataFrame(datos_ejemplo)

    # Guardar a CSV (simulando lo que tienen los alumnos)
    df_original.to_csv('dataset_ppt_ejemplo.csv', index=False)
    print(f"✓ CSV creado: {len(df_original)} rondas")
    print(f"  Columnas: {list(df_original.columns)}")
    print(f"\nPrimeras 5 filas:")
    print(df_original.head())

    # ========== PASO 2: Cargar CSV ==========
    print("\n[PASO 2] Cargando CSV...")

    df = pd.read_csv('dataset_ppt_ejemplo.csv')
    print(f"✓ Cargado: {len(df)} rondas")

    # ========== PASO 3: Generar features para cada ronda ==========
    print("\n[PASO 3] Generando features para cada ronda...")

    fe = PPTFeatureEngineering()

    # Lista para acumular las features de todas las rondas
    todas_las_features = []

    for idx in range(len(df)):
        # Obtener todas las jugadas hasta la ronda actual
        jugadas_jugador_hasta_ahora = df['jugada_jugador'].iloc[:idx+1].tolist()
        jugadas_oponente_hasta_ahora = df['jugada_oponente'].iloc[:idx+1].tolist()
        numero_ronda = df['numero_ronda'].iloc[idx]

        # Generar features
        features = fe.generar_features_completas(
            jugadas_oponente=jugadas_oponente_hasta_ahora,
            jugadas_jugador=jugadas_jugador_hasta_ahora,
            numero_ronda=numero_ronda,
            total_rondas=50  # Estimado
        )

        # Añadir la próxima jugada del oponente como objetivo (si existe)
        if idx < len(df) - 1:
            features['objetivo'] = df['jugada_oponente'].iloc[idx + 1]
        else:
            features['objetivo'] = None

        todas_las_features.append(features)

    print(f"✓ Features generadas para {len(todas_las_features)} rondas")

    # ========== PASO 4: Crear DataFrame con features ==========
    print("\n[PASO 4] Creando DataFrame final...")

    df_features = pd.DataFrame(todas_las_features)

    # Eliminar la última fila (no tiene objetivo)
    df_features = df_features[df_features['objetivo'].notna()]

    print(f"✓ DataFrame final: {df_features.shape[0]} filas × {df_features.shape[1]} columnas")
    print(f"\nPrimeras features:")
    print(df_features.columns[:15].tolist())
    print("...")
    print(df_features.columns[-5:].tolist())

    # ========== PASO 5: Guardar para entrenamiento ==========
    print("\n[PASO 5] Guardando dataset con features...")

    df_features.to_csv('dataset_ppt_con_features.csv', index=False)
    print("✓ Guardado como: dataset_ppt_con_features.csv")

    # ========== PASO 6: Mostrar estadísticas ==========
    print("\n[PASO 6] Estadísticas del dataset final:")
    print("-" * 70)

    print(f"\nDataset original:")
    print(f"  - Columnas: {len(df_original.columns)} (numero_ronda, jugada_jugador, jugada_oponente)")
    print(f"  - Filas: {len(df_original)}")

    print(f"\nDataset con features:")
    print(f"  - Columnas: {len(df_features.columns)} ({len(df_features.columns)-1} features + 1 objetivo)")
    print(f"  - Filas: {len(df_features)} (perdemos última ronda sin objetivo)")

    print(f"\nEjemplos de features generadas:")
    feature_categories = {
        'Frecuencia': [c for c in df_features.columns if 'freq' in c][:3],
        'Patrones': [c for c in df_features.columns if 'lag' in c or 'patron' in c][:3],
        'Rachas': [c for c in df_features.columns if 'racha' in c][:3],
        'Temporales': [c for c in df_features.columns if 'fase' in c or 'progreso' in c][:3],
        'Entropía': [c for c in df_features.columns if 'entropia' in c][:2],
        'Markov': [c for c in df_features.columns if 'markov' in c][:3],
    }

    for categoria, features in feature_categories.items():
        print(f"\n  {categoria}:")
        for feat in features:
            print(f"    - {feat}")

    # ========== PASO 7: Ejemplo de uso para ML ==========
    print("\n[PASO 7] Preparación para Machine Learning:")
    print("-" * 70)

    # Separar features y objetivo
    X = df_features.drop(columns=['objetivo', 'patron_2_string', 'patron_3_string'])
    # (eliminamos string features que necesitan encoding adicional)

    y = df_features['objetivo']

    print(f"\nFeatures (X): {X.shape}")
    print(f"Objetivo (y): {y.shape}")
    print(f"\nDistribución del objetivo:")
    print(y.value_counts())

    print("\n" + "=" * 70)
    print("¡Listo! Ya tienes un dataset preparado para entrenar modelos de ML")
    print("=" * 70)

    print("\nPróximos pasos:")
    print("1. Cargar 'dataset_ppt_con_features.csv'")
    print("2. Split temporal: train (primeros 70%), test (últimos 30%)")
    print("3. Entrenar modelos: RandomForest, XGBoost, etc.")
    print("4. Evaluar accuracy y matriz de confusión")
    print("5. Seleccionar mejores features y reentrenar")

    return df_features


def ejemplo_uso_incremental():
    """Muestra cómo usar las features en tiempo real (incremental)"""

    print("\n\n" + "=" * 70)
    print("EJEMPLO 2: Uso Incremental (predicción en cada ronda)")
    print("=" * 70)

    fe = PPTFeatureEngineering()

    # Simular un juego en progreso
    jugadas_jugador = []
    jugadas_oponente = []

    print("\nSimulando un juego...")

    partidas = [
        ("piedra", "papel"),
        ("papel", "piedra"),
        ("piedra", "tijera"),
        ("tijera", "tijera"),
        ("papel", "piedra"),
    ]

    for ronda, (j_jugador, j_oponente) in enumerate(partidas, start=1):
        jugadas_jugador.append(j_jugador)
        jugadas_oponente.append(j_oponente)

        # Calcular resultado
        resultado = fe.calcular_resultado(j_jugador, j_oponente)

        # Generar features hasta ahora
        features = fe.generar_features_completas(
            jugadas_oponente=jugadas_oponente,
            jugadas_jugador=jugadas_jugador,
            numero_ronda=ronda
        )

        print(f"\n--- Ronda {ronda} ---")
        print(f"  Jugadas: {j_jugador} vs {j_oponente} → {resultado}")
        print(f"  Features generadas: {len(features)}")
        print(f"  Frecuencia piedra (oponente): {features['freq_global_piedra']:.2f}")
        print(f"  Frecuencia papel (oponente): {features['freq_global_papel']:.2f}")
        print(f"  Frecuencia tijera (oponente): {features['freq_global_tijera']:.2f}")
        print(f"  Entropía (predictibilidad): {features['entropia_global']:.2f}")

        # Aquí harías la predicción con tu modelo
        # prediccion = modelo.predict([features])


if __name__ == "__main__":
    print("\n" * 2)
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "FEATURE ENGINEERING PARA PPT" + " " * 25 + "#")
    print("#" + " " * 20 + "Ejemplo de Uso Completo" + " " * 25 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Ejecutar ejemplos
    df_features = ejemplo_completo_desde_csv()
    ejemplo_uso_incremental()

    print("\n\n" + "=" * 70)
    print("Archivos generados:")
    print("  - dataset_ppt_ejemplo.csv (datos originales)")
    print("  - dataset_ppt_con_features.csv (listo para ML)")
    print("=" * 70 + "\n")
