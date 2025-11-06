"""
EJERCICIO COMPLETO: Entrenar Modelo para Piedra, Papel o Tijera
================================================================

OBJETIVO:
Entrenar un modelo de Machine Learning que prediga la próxima jugada del oponente
en el juego Piedra, Papel o Tijera.

FLUJO COMPLETO:
1. Cargar datos históricos de partidas
2. Generar features (usando lo aprendido en Clase 06)
3. Preparar datos (train/test split)
4. Entrenar múltiples modelos
5. Evaluar y comparar
6. Seleccionar el mejor modelo
7. Implementar estrategia de juego

IMPORTANTE: Este ejercicio tiene versión ALUMNO (para completar) y
una versión de DEMOSTRACIÓN completa.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# PARTE 1: FEATURE ENGINEERING (Repaso de Clase 06)
# =============================================================================

class PPTFeatureGenerator:
    """Genera features para el modelo de PPT"""

    def __init__(self):
        self.jugadas_counter = {
            "piedra": "papel",
            "papel": "tijera",
            "tijera": "piedra"
        }

    def calcular_resultado(self, jugada_jugador, jugada_oponente):
        """Calcula el resultado de una ronda"""
        if jugada_jugador == jugada_oponente:
            return "empate"

        gana = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
        return "victoria" if gana[jugada_jugador] == jugada_oponente else "derrota"

    def generar_features_basicas(self, historial_oponente, historial_jugador, numero_ronda):
        """
        Genera features básicas para una ronda

        Args:
            historial_oponente: lista de jugadas del oponente hasta ahora
            historial_jugador: lista de jugadas del jugador hasta ahora
            numero_ronda: número de ronda actual

        Returns:
            dict con features
        """
        features = {}

        # Features de frecuencia global
        if historial_oponente:
            total = len(historial_oponente)
            conteo = Counter(historial_oponente)

            features['freq_piedra'] = conteo.get('piedra', 0) / total
            features['freq_papel'] = conteo.get('papel', 0) / total
            features['freq_tijera'] = conteo.get('tijera', 0) / total
        else:
            features['freq_piedra'] = 0.33
            features['freq_papel'] = 0.33
            features['freq_tijera'] = 0.33

        # Features de frecuencia reciente (últimas 5 jugadas)
        if len(historial_oponente) >= 5:
            reciente = historial_oponente[-5:]
            total_rec = len(reciente)
            conteo_rec = Counter(reciente)

            features['freq_5_piedra'] = conteo_rec.get('piedra', 0) / total_rec
            features['freq_5_papel'] = conteo_rec.get('papel', 0) / total_rec
            features['freq_5_tijera'] = conteo_rec.get('tijera', 0) / total_rec
        else:
            features['freq_5_piedra'] = features['freq_piedra']
            features['freq_5_papel'] = features['freq_papel']
            features['freq_5_tijera'] = features['freq_tijera']

        # Última jugada (lag_1) - One-hot encoding
        if historial_oponente:
            ultima = historial_oponente[-1]
            features['lag_1_piedra'] = 1 if ultima == 'piedra' else 0
            features['lag_1_papel'] = 1 if ultima == 'papel' else 0
            features['lag_1_tijera'] = 1 if ultima == 'tijera' else 0
        else:
            features['lag_1_piedra'] = 0
            features['lag_1_papel'] = 0
            features['lag_1_tijera'] = 0

        # Penúltima jugada (lag_2)
        if len(historial_oponente) >= 2:
            penultima = historial_oponente[-2]
            features['lag_2_piedra'] = 1 if penultima == 'piedra' else 0
            features['lag_2_papel'] = 1 if penultima == 'papel' else 0
            features['lag_2_tijera'] = 1 if penultima == 'tijera' else 0
        else:
            features['lag_2_piedra'] = 0
            features['lag_2_papel'] = 0
            features['lag_2_tijera'] = 0

        # Rachas de victorias/derrotas
        if historial_oponente and historial_jugador:
            resultados = [
                self.calcular_resultado(j_jug, j_op)
                for j_jug, j_op in zip(historial_jugador, historial_oponente)
            ]

            # Racha actual
            racha_actual = 1
            if resultados:
                ultimo_resultado = resultados[-1]
                for i in range(len(resultados) - 2, -1, -1):
                    if resultados[i] == ultimo_resultado:
                        racha_actual += 1
                    else:
                        break

                features['racha_victorias'] = racha_actual if ultimo_resultado == 'victoria' else 0
                features['racha_derrotas'] = racha_actual if ultimo_resultado == 'derrota' else 0
            else:
                features['racha_victorias'] = 0
                features['racha_derrotas'] = 0
        else:
            features['racha_victorias'] = 0
            features['racha_derrotas'] = 0

        # Features temporales
        features['numero_ronda'] = numero_ronda
        features['fase_inicio'] = 1 if numero_ronda < 10 else 0
        features['fase_medio'] = 1 if 10 <= numero_ronda < 30 else 0
        features['fase_final'] = 1 if numero_ronda >= 30 else 0

        return features


# =============================================================================
# PARTE 2: PREPARACIÓN DE DATOS
# =============================================================================

def cargar_y_preparar_datos(archivo_csv='datos_ppt_ejemplo.csv'):
    """
    Carga datos de partidas y genera features

    El CSV debe tener al menos estas columnas:
    - jugada_jugador
    - jugada_oponente
    - numero_ronda
    """
    print("=" * 70)
    print("PASO 1: CARGAR Y PREPARAR DATOS")
    print("=" * 70)

    # Cargar CSV
    try:
        df = pd.read_csv(archivo_csv)
        print(f"\n Dataset cargado: {len(df)} rondas")
    except FileNotFoundError:
        print(f"\n No se encontró {archivo_csv}")
        print("  Generando datos sintéticos para demostración...\n")
        df = generar_datos_sinteticos()

    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nPrimeras filas:")
    print(df.head())

    # Generar features
    print("\n Generando features...")
    feature_gen = PPTFeatureGenerator()

    lista_features = []
    lista_targets = []

    for idx in range(len(df)):
        # Historial hasta esta ronda (NO incluye la ronda actual)
        hist_oponente = df['jugada_oponente'][:idx].tolist()
        hist_jugador = df['jugada_jugador'][:idx].tolist()
        numero_ronda = df.loc[idx, 'numero_ronda']

        # Target: la jugada actual del oponente (lo que queremos predecir)
        target = df.loc[idx, 'jugada_oponente']

        # Generar features basadas en el historial
        if idx > 0:  # Necesitamos al menos 1 jugada de historial
            features = feature_gen.generar_features_basicas(
                hist_oponente, hist_jugador, numero_ronda
            )
            lista_features.append(features)
            lista_targets.append(target)

    # Crear DataFrames
    X = pd.DataFrame(lista_features)
    y = pd.Series(lista_targets)

    print(f"\n Features generadas: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"\nNombres de features:")
    for col in X.columns:
        print(f"  - {col}")

    print(f"\nDistribución de clases (target):")
    print(y.value_counts())

    return X, y


def generar_datos_sinteticos(n_rondas=200):
    """Genera datos sintéticos para demostración"""
    np.random.seed(42)

    # Simular oponente con patrón:
    # - Prefiere piedra (40%)
    # - Luego papel (35%)
    # - Luego tijera (25%)

    jugadas_oponente = np.random.choice(
        ['piedra', 'papel', 'tijera'],
        size=n_rondas,
        p=[0.40, 0.35, 0.25]
    )

    # Simular jugador con estrategia básica
    jugadas_jugador = []
    for jug_op in jugadas_oponente:
        # Counter strategy con algo de ruido
        counter = {'piedra': 'papel', 'papel': 'tijera', 'tijera': 'piedra'}
        if np.random.random() < 0.7:  # 70% usa counter
            jugadas_jugador.append(counter[jug_op])
        else:  # 30% aleatorio
            jugadas_jugador.append(np.random.choice(['piedra', 'papel', 'tijera']))

    df = pd.DataFrame({
        'numero_ronda': range(1, n_rondas + 1),
        'jugada_jugador': jugadas_jugador,
        'jugada_oponente': jugadas_oponente
    })

    return df


# =============================================================================
# PARTE 3: ENTRENAR MODELOS
# =============================================================================

def entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test):
    """Entrena múltiples modelos y los compara"""
    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR Y COMPARAR MODELOS")
    print("=" * 70)

    # Definir modelos
    modelos = {
        'KNN (K=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (K=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }

    print(f"\nEntrenando {len(modelos)} modelos...\n")

    resultados = {}

    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        # Evaluar
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'modelo': modelo,
            'acc_train': acc_train,
            'acc_test': acc_test,
            'y_pred': y_pred_test
        }

    # Mostrar resultados
    print("Resultados:")
    print(f"{'Modelo':<30} {'Train':>8} {'Test':>8} {'Diferencia':>12}")
    print("-" * 70)

    baseline = 1.0 / 3.0  # Aleatorio: 33.3%

    for nombre, res in sorted(resultados.items(), key=lambda x: x[1]['acc_test'], reverse=True):
        diff = abs(res['acc_train'] - res['acc_test'])
        simbolo = "" if res['acc_test'] > baseline else " "

        print(f"{simbolo} {nombre:<28} {res['acc_train']:>7.1%} {res['acc_test']:>7.1%} "
              f"{diff:>11.1%}")

    print("-" * 70)
    print(f"{'Baseline (Aleatorio)':<30} {'':>8} {baseline:>7.1%}")

    # Mejor modelo
    mejor = max(resultados.items(), key=lambda x: x[1]['acc_test'])
    print(f"\n MEJOR MODELO: {mejor[0]}")
    print(f"  Accuracy en test: {mejor[1]['acc_test']:.1%}")

    mejora = (mejor[1]['acc_test'] - baseline) / baseline * 100
    print(f"  Mejora sobre baseline: +{mejora:.1f}%")

    return resultados, mejor


# =============================================================================
# PARTE 4: EVALUACIÓN DETALLADA
# =============================================================================

def evaluar_mejor_modelo(nombre_modelo, resultado, y_test):
    """Evaluación detallada del mejor modelo"""
    print("\n" + "=" * 70)
    print(f"PASO 3: EVALUACIÓN DETALLADA - {nombre_modelo}")
    print("=" * 70)

    y_pred = resultado['y_pred']

    # Accuracy
    print(f"\n1. ACCURACY")
    print("-" * 70)
    print(f"  Train: {resultado['acc_train']:.2%}")
    print(f"  Test:  {resultado['acc_test']:.2%}")

    # Confusion Matrix
    print(f"\n2. MATRIZ DE CONFUSIÓN")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=['piedra', 'papel', 'tijera'])

    print("\n                 PREDICHO")
    print("           piedra  papel  tijera")
    print(f"REAL piedra   {cm[0][0]:3d}    {cm[0][1]:3d}    {cm[0][2]:3d}")
    print(f"     papel    {cm[1][0]:3d}    {cm[1][1]:3d}    {cm[1][2]:3d}")
    print(f"     tijera   {cm[2][0]:3d}    {cm[2][1]:3d}    {cm[2][2]:3d}")

    # Análisis de confusión
    print("\n Análisis:")
    for i, jugada_real in enumerate(['piedra', 'papel', 'tijera']):
        total = sum(cm[i])
        correctos = cm[i][i]
        print(f"  {jugada_real:7s}: {correctos}/{total} correctos ({correctos/total:.1%})")

    # Classification Report
    print(f"\n3. CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_test, y_pred))

    # Feature importance (si es árbol o RF)
    modelo = resultado['modelo']
    if hasattr(modelo, 'feature_importances_'):
        print(f"\n4. IMPORTANCIA DE FEATURES")
        print("-" * 70)
        importancias = modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]

        print("\nTop 10 features más importantes:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {modelo.feature_names_in_[idx]:25s}: {importancias[idx]:.4f}")


# =============================================================================
# PARTE 5: IMPLEMENTAR ESTRATEGIA DE JUEGO
# =============================================================================

class JugadorIA:
    """Jugador de PPT con IA"""

    def __init__(self, modelo, feature_generator):
        self.modelo = modelo
        self.feature_gen = feature_generator
        self.historial_oponente = []
        self.historial_propio = []
        self.numero_ronda = 0

        self.counter = {
            'piedra': 'papel',
            'papel': 'tijera',
            'tijera': 'piedra'
        }

    def predecir_y_jugar(self):
        """
        Predice la próxima jugada del oponente y devuelve el counter

        Returns:
            str: jugada a realizar ('piedra', 'papel', o 'tijera')
        """
        self.numero_ronda += 1

        # Primera ronda: jugar aleatorio
        if self.numero_ronda == 1:
            return np.random.choice(['piedra', 'papel', 'tijera'])

        # Generar features
        features = self.feature_gen.generar_features_basicas(
            self.historial_oponente,
            self.historial_propio,
            self.numero_ronda
        )

        # Convertir a DataFrame
        X = pd.DataFrame([features])

        # Predecir jugada del oponente
        prediccion_oponente = self.modelo.predict(X)[0]

        # Jugar el counter
        mi_jugada = self.counter[prediccion_oponente]

        return mi_jugada

    def registrar_resultado(self, mi_jugada, jugada_oponente):
        """Registra el resultado de una ronda"""
        self.historial_propio.append(mi_jugada)
        self.historial_oponente.append(jugada_oponente)


def simular_partida(jugador_ia, oponente_patron, n_rondas=50):
    """Simula una partida completa"""
    print("\n" + "=" * 70)
    print("PASO 4: SIMULAR PARTIDA CON IA")
    print("=" * 70)

    print(f"\nSimulando {n_rondas} rondas...")
    print("  Jugador IA vs Oponente con patrón (40% piedra, 35% papel, 25% tijera)")

    victorias = 0
    derrotas = 0
    empates = 0

    gana = {'piedra': 'tijera', 'papel': 'piedra', 'tijera': 'papel'}

    for ronda in range(n_rondas):
        # IA juega
        jugada_ia = jugador_ia.predecir_y_jugar()

        # Oponente juega (según patrón)
        jugada_oponente = np.random.choice(
            ['piedra', 'papel', 'tijera'],
            p=[0.40, 0.35, 0.25]
        )

        # Calcular resultado
        if jugada_ia == jugada_oponente:
            resultado = 'empate'
            empates += 1
        elif gana[jugada_ia] == jugada_oponente:
            resultado = 'victoria'
            victorias += 1
        else:
            resultado = 'derrota'
            derrotas += 1

        # Registrar
        jugador_ia.registrar_resultado(jugada_ia, jugada_oponente)

        # Mostrar algunas rondas
        if ronda < 5 or ronda >= n_rondas - 5:
            print(f"  Ronda {ronda+1:2d}: IA={jugada_ia:7s}, Oponente={jugada_oponente:7s} → {resultado}")
        elif ronda == 5:
            print("  ...")

    # Resultados finales
    print(f"\nResultados finales:")
    print(f"  Victorias: {victorias} ({victorias/n_rondas:.1%})")
    print(f"  Derrotas:  {derrotas} ({derrotas/n_rondas:.1%})")
    print(f"  Empates:   {empates} ({empates/n_rondas:.1%})")

    winrate = victorias / n_rondas
    print(f"\n Win Rate: {winrate:.1%}")

    if winrate > 0.50:
        print(" ¡Excelente! La IA está ganando más de lo esperado")
    elif winrate > 0.40:
        print(" Bueno! La IA está aprovechando el patrón del oponente")
    elif winrate > 0.33:
        print(" La IA está ligeramente mejor que aleatorio")
    else:
        print(" La IA no está superando la estrategia aleatoria")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Flujo completo del ejercicio"""
    print("\n" + "" * 35)
    print("EJERCICIO COMPLETO: ENTRENAR MODELO PARA PPT")
    print("" * 35)

    # PASO 1: Cargar y preparar datos
    X, y = cargar_y_preparar_datos()

    # Train/test split
    print("\n División train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Mantener proporción de clases
    )

    print(f"  Train: {len(X_train)} muestras")
    print(f"  Test:  {len(X_test)} muestras")

    # PASO 2: Entrenar y comparar modelos
    resultados, (nombre_mejor, mejor) = entrenar_y_comparar_modelos(
        X_train, X_test, y_train, y_test
    )

    # PASO 3: Evaluación detallada
    evaluar_mejor_modelo(nombre_mejor, mejor, y_test)

    # PASO 4: Simular partida
    feature_gen = PPTFeatureGenerator()
    jugador_ia = JugadorIA(mejor['modelo'], feature_gen)
    simular_partida(jugador_ia, 'patron', n_rondas=50)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN Y CONCLUSIONES")
    print("=" * 70)

    print(f"\n MEJOR MODELO: {nombre_mejor}")
    print(f"  Accuracy: {mejor['acc_test']:.1%}")
    print(f"  Mejora sobre aleatorio: +{(mejor['acc_test'] - 0.333) / 0.333 * 100:.1f}%")

    print(f"\n PRÓXIMOS PASOS:")
    print("  1. Experimentar con más features (entropía, markov, etc.)")
    print("  2. Probar otros algoritmos (SVM, Neural Networks)")
    print("  3. Ajustar hiperparámetros (GridSearch)")
    print("  4. Validar contra oponentes humanos reales")
    print("  5. Implementar en app interactiva")

    print("\n" + "" * 35)


if __name__ == "__main__":
    main()
