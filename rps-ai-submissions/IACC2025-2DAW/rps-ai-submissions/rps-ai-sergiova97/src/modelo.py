"""
EJERCICIO PARA ALUMNOS: Entrenar Modelo para Piedra, Papel o Tijera
====================================================================

OBJETIVO:
Aplicar TODO lo aprendido en esta clase para crear un modelo que prediga
la próxima jugada del oponente en PPT usando tus DATOS REALES de partidas.

REQUISITOS PREVIOS:
- Debes tener un CSV con datos de tus partidas contra compañeros
- El CSV debe tener al menos estas columnas:
  * jugada_jugador (tu jugada: piedra/papel/tijera)
  * jugada_oponente (jugada del oponente: piedra/papel/tijera)
  * numero_ronda (número de ronda: 1, 2, 3, ...)

TAREAS:
1. Usar tus datos reales de partidas (mínimo 100 rondas)
2. Crear función para generar features básicas
3. Preparar datos con train/test split
4. Entrenar múltiples modelos (KNN, Decision Tree, Random Forest)
5. Comparar resultados
6. Implementar un jugador IA que use el mejor modelo

OBJETIVO:
- Lograr >50% accuracy (mejor que aleatorio 33%)
- Comparar al menos 3 modelos diferentes
- Mostrar confusion matrix
- (Bonus) Implementar clase JugadorIA

¡BUENA SUERTE!
"""
import time
from pathlib import Path
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado_final.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}
# =============================================================================
# PARTE 1: GENERAR FEATURES
# =============================================================================

class PPTFeatureGenerator:
    """
    Genera features para el modelo de PPT
    """

    def calcular_resultado(self, jugada_jugador, jugada_oponente):
        """
        Calcula el resultado de una ronda

        Debe retornar: "victoria", "derrota", o "empate"
        """

        resultado = "victoria"

        if jugada_jugador == jugada_oponente:
            resultado = "empate"

        if PIERDE_CONTRA[jugada_jugador] == jugada_oponente:
            resultado = "derrota"

        return resultado

    def generar_features_basicas(self, historial_oponente, historial_jugador, historial_oponente_global, historial_jugador_global, historial_partida, historial_global, numero_ronda, tiempo1, tiempo2):
        """
        Genera features básicas para una ronda

        Args:
            historial_oponente: lista de jugadas del oponente hasta ahora
            historial_jugador: lista de jugadas del jugador hasta ahora
            numero_ronda: número de ronda actual

        Returns:
            dict con features

        - freq_piedra, freq_papel, freq_tijera (frecuencia global)
        - freq_5_piedra, freq_5_papel, freq_5_tijera (últimas 5 jugadas)
        - lag_1_piedra, lag_1_papel, lag_1_tijera (última jugada, one-hot)
        - lag_2_* (penúltima jugada)
        - racha_victorias, racha_derrotas
        - numero_ronda
        - fase_inicio, fase_medio, fase_final (one-hot)

        PISTA: Revisa los ejercicios de Feature Engineering (Clase 06)
        """
        features = {
            'numero_ronda': 0,
            #'fase_inicio': 0,
            #'fase_medio': 0,
            #'fase_final': 0,
            'reaccion_rapida_jugador1': 0,
            'reaccion_media_jugador1': 0,
            'reaccion_lenta_jugador1': 0,
            'reaccion_rapida_jugador2': 0,
            'reaccion_media_jugador2': 0,
            'reaccion_lenta_jugador2': 0,
        }

        self.generar_features_frecuencias_partida(features, historial_jugador, historial_oponente, historial_partida)
        self.generar_features_frecuencias_global(features, historial_jugador_global, historial_oponente_global, historial_global)
        self.generar_features_last5(features, historial_jugador, historial_oponente)
        self.generar_features_lag(features, historial_jugador, historial_oponente)
        self.generar_features_rachas(features, historial_jugador, historial_oponente)

        # Implementa features temporales
        features['numero_ronda'] = numero_ronda

        # fase_inicio, fase_medio, fase_final
        #if numero_ronda <= 5:
        #    features['fase_inicio'] = 1
        #elif numero_ronda <= 15:
        #    features['fase_medio'] = 1
        #else:
        #    features['fase_final'] = 1

        # otras features
        self.generar_features_mantener_victoria(features, historial_jugador, historial_oponente)
        self.generar_features_cambio_derrota(features, historial_jugador, historial_oponente)
        self.generar_features_jugar_contra(features, historial_jugador, historial_oponente)

        if tiempo1 < 2:
            features['reaccion_rapida_jugador1'] = 1
        elif tiempo1 <= 4.45:
            features['reaccion_media_jugador1'] = 1
        else:
            features['reaccion_lenta_jugador1'] = 1

        if tiempo2 < 2:
            features['reaccion_rapida_jugador2'] = 1
        elif tiempo2 <= 4.45:
            features['reaccion_media_jugador2'] = 1
        else:
            features['reaccion_lenta_jugador2'] = 1

        return features

    def generar_features_frecuencias_partida(self, features, historial_jugador, historial_oponente, historial_partida):

        if historial_jugador:
            total = len(historial_jugador)

            features['freq_piedra_jugador1_partida'] = historial_jugador.count('piedra') / total
            features['freq_papel_jugador1_partida'] = historial_jugador.count('papel') / total
            features['freq_tijera_jugador1_partida'] = historial_jugador.count('tijera') / total
        else:
            features['freq_piedra_jugador1_partida'] = 0.33
            features['freq_papel_jugador1_partida'] = 0.33
            features['freq_tijera_jugador1_partida'] = 0.33

        if historial_oponente:
            total = len(historial_oponente)

            features['freq_piedra_jugador2_partida'] = historial_oponente.count('piedra') / total
            features['freq_papel_jugador2_partida'] = historial_oponente.count('papel') / total
            features['freq_tijera_jugador2_partida'] = historial_oponente.count('tijera') / total
        else:
            features['freq_piedra_jugador2_partida'] = 0.33
            features['freq_papel_jugador2_partida'] = 0.33
            features['freq_tijera_jugador2_partida'] = 0.33

        if historial_partida:
            total = len(historial_partida)

            features['freq_piedra_partida'] = historial_partida.count('piedra') / total
            features['freq_papel_partida'] = historial_partida.count('papel') / total
            features['freq_tijera_partida'] = historial_partida.count('tijera') / total
        else:
            features['freq_piedra_partida'] = 0.33
            features['freq_papel_partida'] = 0.33
            features['freq_tijera_partida'] = 0.33

    def generar_features_frecuencias_global(self, features, historial_jugador_global, historial_oponente_global, historial_global):
        if historial_jugador_global:
            total = len(historial_jugador_global)

            features['freq_piedra_jugador1_global'] = historial_jugador_global.count('piedra') / total
            features['freq_papel_jugador1_global'] = historial_jugador_global.count('papel') / total
            features['freq_tijera_jugador1_global'] = historial_jugador_global.count('tijera') / total
        else:
            features['freq_piedra_jugador1_global'] = 0.33
            features['freq_papel_jugador1_global'] = 0.33
            features['freq_tijera_jugador1_global'] = 0.33

        if historial_oponente_global:
            total = len(historial_oponente_global)
            features['freq_piedra_jugador2_global'] = historial_oponente_global.count('piedra') / total
            features['freq_papel_jugador2_global'] = historial_oponente_global.count('papel') / total
            features['freq_tijera_jugador2_global'] = historial_oponente_global.count('tijera') / total
        else:
            features['freq_piedra_jugador2_global'] = 0.33
            features['freq_papel_jugador2_global'] = 0.33
            features['freq_tijera_jugador2_global'] = 0.33

        if historial_global:
            total = len(historial_global)
            features['freq_piedra_global'] = historial_global.count('piedra') / total
            features['freq_papel_global'] = historial_global.count('papel') / total
            features['freq_tijera_global'] = historial_global.count('tijera') / total
        else:
            features['freq_piedra_global'] = 0.33
            features['freq_papel_global'] = 0.33
            features['freq_tijera_global'] = 0.33

    def generar_features_last5(self, features, historial_jugador, historial_oponente):
        N = 5
        historial_jugador1_last5 = historial_jugador[-N:]
        historial_jugador2_last5 = historial_oponente[-N:]

        if historial_jugador1_last5:
            total = len(historial_jugador1_last5)
            features['freq_piedra_jugador1_last5'] = historial_jugador1_last5.count('piedra') / total
            features['freq_papel_jugador1_last5'] = historial_jugador1_last5.count('papel') / total
            features['freq_tijera_jugador1_last5'] = historial_jugador1_last5.count('tijera') / total
        else:
            features['freq_piedra_jugador1_last5'] = 0.33
            features['freq_papel_jugador1_last5'] = 0.33
            features['freq_tijera_jugador1_last5'] = 0.33

        if historial_jugador2_last5:
            total = len(historial_jugador2_last5)
            features['freq_piedra_jugador2_last5'] = historial_jugador2_last5.count('piedra') / total
            features['freq_papel_jugador2_last5'] = historial_jugador2_last5.count('papel') / total
            features['freq_tijera_jugador2_last5'] = historial_jugador2_last5.count('tijera') / total
        else:
            features['freq_piedra_jugador2_last5'] = 0.33
            features['freq_papel_jugador2_last5'] = 0.33
            features['freq_tijera_jugador2_last5'] = 0.33

    def generar_features_lag(self, features, historial_jugador, historial_oponente):
        features['lag_1_jugador1'] = self.safe_lag(historial_jugador, -2)
        features['lag_2_jugador1'] = self.safe_lag(historial_jugador, -3)
        features['lag_1_jugador2'] = self.safe_lag(historial_oponente, -2)
        features['lag_2_jugador2'] = self.safe_lag(historial_oponente, -3)

    def generar_features_rachas(self, features, historial_jugador, historial_oponente):
        rachas = self.sacar_racha(historial_jugador, historial_oponente)
        features['racha_jugador1'] = rachas[0]
        features['racha_jugador2'] = rachas[1]

    def generar_features_mantener_victoria(self, features, historial_jugador, historial_oponente):
        features['prob_mantener_victoria_jugador'] = self.calcular_mantener_victoria(historial_jugador, historial_oponente)
        features['prob_mantener_victoria_jugador2'] = self.calcular_mantener_victoria(historial_oponente, historial_jugador)

    def generar_features_cambio_derrota(self, features, historial_jugador, historial_oponente):
        features['prob_cambio_derrota_jugador'] = self.calcular_cambio_derrota(historial_jugador, historial_oponente)
        features['prob_cambio_derrota_jugador2'] = self.calcular_cambio_derrota(historial_oponente, historial_jugador)

    def generar_features_jugar_contra(self, features, historial_jugador, historial_oponente):
        features['prob_jugar_contra_jugador'] = self.calcular_jugar_contra(historial_jugador, historial_oponente)
        features['prob_jugar_contra_jugador2'] = self.calcular_jugar_contra(historial_oponente, historial_jugador)

    def safe_lag(self, historial, pos):
        ret = 0
        if len(historial) >= abs(pos):
            jugada = self.sacar_jugada(historial, pos, '')
            ret = self.PPTValorNum(jugada) if jugada else 0

        return ret

    def sacar_jugada(self, lst, idx, default=None):
        try:
            return lst[idx]
        except IndexError:
            return default

    def sacar_racha(self, historial_jugador, historial_oponente):
        """
        Determina la racha de victorias de ambos jugadores
        """
        racha_jugador = 0
        racha_oponente = 0

        if len(historial_jugador) > 0:
            racha_jugador = 0
            racha_oponente = 0

            jugadas_previas = list(zip(historial_jugador, historial_oponente))

            for jugador, oponente in jugadas_previas:
                resultado = self.calcular_resultado(jugador, oponente)

                if resultado == 'victoria':
                    racha_jugador += 1
                    racha_oponente = 0
                elif resultado == 'derrota':
                    racha_jugador = 0
                    racha_oponente += 1
                else:
                    racha_jugador = 0
                    racha_oponente = 0

        return racha_jugador, racha_oponente

    def calcular_mantener_victoria(self, historial_jugador, historial_oponente):
        """
        Calcula la probabilidad de que el jugador repita la jugada tras ganar
        0.5 con menos de 2 rondas o 0 victorias
        """
        prob = 0.5

        if len(historial_jugador) >= 2:
            se_mantiene = 0
            victorias = 0

            for i in range (1, len(historial_jugador)):
                jug_prev = historial_jugador[i-1]
                jug_actual = historial_jugador[i]

                oponente_prev = historial_oponente[i-1]
                resultado = self.calcular_resultado(jug_prev, oponente_prev)

                if resultado == 'victoria':
                    victorias += 1
                    if jug_actual == jug_prev:
                        se_mantiene += 1

            if victorias != 0:
                prob = se_mantiene / victorias

        return prob

    def calcular_cambio_derrota(self, historial_jugador, historial_oponente):
        """
        Calcula la probabilidad de que el jugador cambie su elección al perder la ronda
        0.5 con menos de 2 rondas o 0 derrotas
        """

        prob = 0.5

        if len(historial_jugador) >= 2:
            cambia = 0
            derrotas = 0

            for i in range (1, len(historial_jugador)):
                jug_prev = historial_jugador[i - 1]
                jug_actual = historial_jugador[i]

                oponente_prev = historial_oponente[i - 1]
                resultado = self.calcular_resultado(jug_prev, oponente_prev)

                if resultado == 'derrota':
                    derrotas += 1
                    if jug_actual != jug_prev:
                        cambia += 1

            if derrotas != 0:
                prob = cambia / derrotas

        return prob

    def calcular_jugar_contra(self, historial_jugador, historial_oponente):
        """
        Calcula la probabilidad de que el jugador responda con la jugada que gana a la última jugada del oponente
        0.33 con menos de 2 rondas o 0 derrotas
        """

        prob = 0.33

        if len(historial_jugador) >= 2:
            contra = 0
            derrotas = 0

            for i in range (1, len(historial_jugador)):
                jug_prev = historial_jugador[i - 1]
                oponente_prev = historial_oponente[i - 1]

                resultado = self.calcular_resultado(jug_prev, oponente_prev)

                if resultado == 'derrota':
                    derrotas += 1
                    jugada_ganadora = self.jugada_que_gana_a(oponente_prev)
                    jug_actual = historial_jugador[i]

                    if jug_actual == jugada_ganadora:
                        contra += 1

            if derrotas != 0:
                prob = contra / derrotas

        return prob

    def jugada_que_gana_a(self, jugada):
        gana = ""
        if jugada == "piedra" : gana = "papel"
        if jugada == "papel" : gana = "tijera"
        if jugada == "tijera" : gana = "piedra"

        return gana

    def PPTValorNum(self, jugada_str):
        jugada = 0
        if jugada_str == "piedra" : jugada = 0
        if jugada_str == "papel" : jugada = 1
        if jugada_str == "tijera" : jugada = 2

        return jugada

    def PPTValorString(self, jugada_num):
        jugada = ""
        if jugada_num == 0 : jugada = "piedra"
        if jugada_num == 1 : jugada = "papel"
        if jugada_num == 2 : jugada = "tijera"

        return jugada

    def PPTResultado(self, resultado_str):
        resultado = 0
        if resultado_str == "victoria" : resultado = 1
        if resultado_str == "derrota" : resultado = -1
        return resultado

# =============================================================================
# PARTE 2: PREPARAR DATOS
# =============================================================================

def cargar_y_preparar_datos(archivo_csv):
    """
    Carga datos de partidas y genera features

    Args:
        archivo_csv: Ruta a TU archivo CSV con datos de partidas
                     Debe tener columnas: jugada_jugador, jugada_oponente, numero_ronda

    1. Cargar CSV con pandas
    2. Verificar que tiene las columnas necesarias
    3. Para cada ronda, generar features basadas en historial previo
    4. Target = jugada actual del oponente
    5. Retornar X (features), y (target)

    IMPORTANTE: Usa solo el historial HASTA cada ronda (no hacer trampa)
    """
    print("=" * 70)
    print("PASO 1: CARGAR Y PREPARAR DATOS")
    print("=" * 70)

    df = pd.read_csv(archivo_csv)

    columnas_necesarias = ['ronda', 'jugada1', 'jugada2', 'ganador', 'tiempo1', 'tiempo2']
    if not all(col in df.columns for col in columnas_necesarias):
        print(f"ERROR: El CSV debe tener columnas: {columnas_necesarias}")
        return None, None

    df['jugada1'] = df['jugada1'].str.lower()
    df['jugada2'] = df['jugada2'].str.lower()

    # TU CÓDIGO AQUÍ
    feature_gen = PPTFeatureGenerator()

    historial_jugador = []
    historial_oponente = []
    historial_jugador_global = []
    historial_oponente_global = []
    historial = []
    historial_global = []

    lista_features = []
    lista_targets = []

    for index, row in df.iterrows():
        numero_ronda = row['ronda']
        tiempo1 = row['tiempo1']
        tiempo2 = row['tiempo2']

        if numero_ronda == 1:
            historial_jugador = []
            historial_oponente = []
            historial = []

        features = feature_gen.generar_features_basicas(historial_oponente, historial_jugador, historial_oponente_global, historial_jugador_global, historial, historial_global, numero_ronda, tiempo1, tiempo2)
        lista_features.append(features)
        lista_targets.append(row['jugada2'])

        historial_jugador.append(row['jugada1'])
        historial_oponente.append(row['jugada2'])
        historial_jugador_global.append(row['jugada1'])
        historial_oponente_global.append(row['jugada2'])
        historial.append(row['jugada1'])
        historial.append(row['jugada2'])
        historial_global.append(row['jugada1'])
        historial_global.append(row['jugada2'])

    X = pd.DataFrame(lista_features)
    y = pd.Series(lista_targets)

    print(f"\n✓ Features generadas: {X.shape[0]} muestras, {X.shape[1]} features")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAR MODELOS
# =============================================================================

def entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test, start_idx, end_idx):
    """
    Entrena múltiples modelos y los compara

    1. Definir diccionario con al menos 3 modelos
    2. Entrenar cada modelo
    3. Evaluar en train y test
    4. Mostrar tabla comparativa
    5. Retornar el mejor modelo
    """
    modelos = {
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, max_features= 0.7, random_state=42)
    }

    resultados_ventana = []

    print(f"\n=== Ventana {start_idx} - {end_idx} ===")

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        score = modelo.score(X_test, y_test)

        resultados_ventana.append({
            'nombre': nombre,
            'modelo': modelo,
            'accuracy': score,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'X_test': X_test,
            'y_test': y_test
        })

        print(f"{nombre:<15} | Accuracy: {score:.4f}")

    return resultados_ventana


def circular_rolling_split(X, y, train_pct=0.8, step=10):
    n = len(X)
    train_size = int(n * train_pct)
    test_size = n - train_size

    splits = []
    start = 0

    # Necesitamos recorrer todo el dataset en pasos de "step"
    for _ in range(n // step):

        # TRAIN circular
        train_indices = [(start + i) % n for i in range(train_size)]
        # TEST circular
        test_indices = [(start + train_size + i) % n for i in range(test_size)]

        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        start_idx = train_indices[0]
        end_idx = train_indices[-1]

        splits.append((X_train, X_test, y_train, y_test, start_idx, end_idx))

        start = (start + step) % n  # mover ventana circularmente

    return splits

# =============================================================================
# PARTE 4: EVALUACIÓN DETALLADA
# =============================================================================

def evaluar_mejor_modelo(nombre_modelo, modelo, X_test, y_test):
    """
    Evaluación detallada del mejor modelo

    TODO: Implementa esta función
    1. Hacer predicciones en test
    2. Calcular accuracy
    3. Mostrar confusion matrix
    4. Mostrar classification report
    5. (Bonus) Mostrar feature importance si es árbol/RF
    """
    print("\n" + "=" * 70)
    print(f"PASO 3: EVALUACIÓN DETALLADA - {nombre_modelo}")
    print("=" * 70)

    # 1. Predicciones
    y_pred = modelo.predict(X_test)

    # 2. Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy del modelo: {acc:.4f}")

    # 3. Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 4. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Feature Importance (solo si el modelo la tiene)
    if hasattr(modelo, "feature_importances_"):
        importances = pd.Series(modelo.feature_importances_, index=X_test.columns)
        importances = importances.sort_values(ascending=False)

        print("\nFeature Importances (Top 15):")
        print(importances.head(15))

    print("=" * 70 + "\n")

def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")

def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)

# =============================================================================
# PARTE 5: JUGADOR IA (BONUS)
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial_jugador = []
        self.historial_oponente = []
        self.historial = []
        self.historial_jugador_global = []
        self.historial_oponente_global = []
        self.historial_global = []

        self.numero_ronda = 0
        self.tiempo1 = 0
        self.tiempo2 = 0

        self.feature_gen = PPTFeatureGenerator()

        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str, tiempo_j1: float, tiempo_j2: float):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
            tiempo_j1: Tiempo del jugador 1
            tiempo_j2: Tiempo del jugador 2
        """
        self.historial_jugador.append(jugada_j1)
        self.historial_oponente.append(jugada_j2)

        self.historial_jugador_global.append(jugada_j1)
        self.historial_oponente_global.append(jugada_j2)

        self.historial.append((jugada_j1, jugada_j2))
        self.historial_global.append((jugada_j1, jugada_j2))

        self.tiempo1 = tiempo_j1
        self.tiempo2 = tiempo_j2

        self.numero_ronda += 1

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion
        """
        features = self.feature_gen.generar_features_basicas(
            self.historial_oponente,
            self.historial_jugador,
            self.historial_oponente_global,
            self.historial_jugador_global,
            self.historial,
            self.historial_global,
            self.numero_ronda,
            self.tiempo1,
            self.tiempo2
        )

        return np.array(list(features.values()), dtype=float)

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        X = self.obtener_features_actuales().reshape(1, -1)
        pred = self.modelo.predict(X)[0]

        return pred

    def decidir_jugada(self) -> tuple[str, float]:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        start_time = time.time()
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            current_time = time.time()
            return np.random.choice(["piedra", "papel", "tijera"]), round(current_time - start_time, 2)

        # Juega lo que le gana a la prediccion
        current_time = time.time()
        return PIERDE_CONTRA[prediccion_oponente], round(current_time - start_time, 2)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Flujo completo del ejercicio

    1. Especificar ruta a TU archivo CSV con datos reales
    2. Cargar y preparar datos
    3. Train/test split
    4. Entrenar y comparar modelos
    5. Evaluación detallada
    6. (Bonus) Simular partida con JugadorIA
    """
    print("\n" + "█" * 35)
    print("EJERCICIO: ENTRENAR MODELO PARA PPT")
    print("█" * 35)

    # Ejemplo: archivo_csv = 'mis_partidas_ppt.csv'
    archivo_csv = '../data/general1vs1.csv'
    # archivo_csv = '../data/conjuntoIgual.csv'
    # archivo_csv = '../data/conjuntoTotalIgual.csv'
    # archivo_csv = '../data/conjuntoTotalMaximo.csv'
    X, y = cargar_y_preparar_datos(archivo_csv)
    splits = circular_rolling_split(X, y)
    todos_resultados = []

    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR Y COMPARAR MODELOS POR VENTANAS")
    print("=" * 70)
    for (X_train, X_test, y_train, y_test, start_idx, end_idx) in splits:
        modelos_ventana = entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test, start_idx, end_idx)
        todos_resultados.extend(modelos_ventana)

    todos_ordenados = sorted(todos_resultados, key=lambda x: x['accuracy'], reverse=True)
    top_10 = todos_ordenados[:10]

    print("\n\n=== TOP 10 MODELOS GLOBAL ===")
    for rank, m in enumerate(top_10, start=1):
        print(f"{rank}. {m['nombre']:<15} | acc={m['accuracy']:.4f} | ventana {m['start_idx']}–{m['end_idx']}")

    mejor = top_10[1]

    evaluar_mejor_modelo(
        mejor['nombre'],
        mejor['modelo'],
        mejor['X_test'],
        mejor['y_test']
    )

    guardar_modelo(mejor['modelo'])

    print("\n" + "█" * 35)
    print("¡EJERCICIO COMPLETADO!")
    print("█" * 35)


if __name__ == "__main__":
    main()

    # print("=" * 70)
    # print("INSTRUCCIONES")
    # print("=" * 70)
    # print("\n1. ANTES DE EMPEZAR:")
    # print("   - Debes tener un CSV con tus partidas (mínimo 100 rondas)")
    # print("   - Si no tienes datos, juega más partidas con compañeros")
    # print("   - Formato CSV necesario:")
    # print("     numero_ronda,jugada_jugador,jugada_oponente")
    # print("     1,piedra,papel")
    # print("     2,tijera,tijera")
    # print("     ...")
    # print("\n2. Implementa todas las funciones marcadas con TODO")
    # print("\n3. Ejecuta este script:")
    # print("   python 04_ejercicio_ppt_ALUMNO.py")
    # print("\n4. OBJETIVO: Lograr >50% accuracy en test")
    # print("\n" + "=" * 70)
    # print("RECURSOS:")
    # print("- Repasa ejemplos 01-03")
    # print("- Revisa Clase 06 (Feature Engineering)")
    # print("- Lee clases/07-entrenamiento-modelos/README.md")
    # print("=" * 70)
