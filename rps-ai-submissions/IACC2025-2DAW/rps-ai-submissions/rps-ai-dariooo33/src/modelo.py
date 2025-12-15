"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera SIN scikit-learn
==================================================================
"""

import os
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "resultado_ppt.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# Configuración
LOOKBACK_WINDOW = 5
PATTERN_LENGTH = 3


# =============================================================================
# MODELO PROPIO SIN SCIKIT-LEARN
# =============================================================================

class MarkovPredictor:
    """Predictor de Markov simple para Piedra, Papel o Tijera."""

    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.history = []

    def fit(self, sequences: List[List[int]]):
        """Entrena el modelo con secuencias de jugadas."""
        for seq in sequences:
            for i in range(len(seq) - self.order):
                # Obtener el patrón de orden 'order'
                pattern = tuple(seq[i:i + self.order])
                next_move = seq[i + self.order]
                # Contar transición
                self.transitions[pattern][next_move] += 1

    def predict(self, recent_pattern: Tuple[int]) -> Optional[int]:
        """Predice el siguiente movimiento dado un patrón reciente."""
        if recent_pattern in self.transitions:
            counter = self.transitions[recent_pattern]
            if counter:
                # Devolver el movimiento más común después de este patrón
                return counter.most_common(1)[0][0]
        return None

    def predict_proba(self, recent_pattern: Tuple[int]) -> List[float]:
        """Devuelve probabilidades para cada jugada."""
        probs = [0.333, 0.333, 0.333]  # Distribución uniforme por defecto

        if recent_pattern in self.transitions:
            counter = self.transitions[recent_pattern]
            total = sum(counter.values())
            if total > 0:
                for move, count in counter.items():
                    probs[move] = count / total

        return probs


class FrequencyPredictor:
    """Predictor basado en frecuencias y patrones simples."""

    def __init__(self):
        self.move_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.history = []
        self.window_size = 10

    def update(self, move: int):
        """Actualiza el historial con un nuevo movimiento."""
        self.history.append(move)
        self.move_counts[move] += 1

        # Mantener solo el tamaño de ventana
        if len(self.history) > self.window_size:
            old_move = self.history.pop(0)
            self.move_counts[old_move] -= 1
            if self.move_counts[old_move] == 0:
                del self.move_counts[old_move]

    def predict(self) -> int:
        """Predice el siguiente movimiento."""
        if not self.history:
            return random.randint(0, 2)

        # Estrategia 1: Frecuencia simple
        if len(self.history) >= 3:
            # Obtener movimiento más común en la ventana
            if self.move_counts:
                most_common = self.move_counts.most_common(1)[0][0]
                # Si repite mucho, asumir que cambiará
                if self.move_counts[most_common] / len(self.history) > 0.6:
                    return (most_common + 1) % 3  # Cambia al siguiente en el ciclo

        # Estrategia 2: Patrones de transición
        if len(self.history) >= 2:
            last_move = self.history[-1]
            # Contar qué sigue después de este movimiento
            next_moves = []
            for i in range(len(self.history) - 1):
                if self.history[i] == last_move and i + 1 < len(self.history):
                    next_moves.append(self.history[i + 1])

            if next_moves:
                counter = Counter(next_moves)
                most_likely = counter.most_common(1)[0][0]
                return most_likely

        # Estrategia 3: Aleatorio sesgado por frecuencia
        total = sum(self.move_counts.values())
        if total > 0:
            probs = [self.move_counts[i] / total for i in range(3)]
            # Invertir probabilidades (jugar contra lo más probable)
            inverted_probs = [(1 - p + 0.1) for p in probs]  # +0.1 para evitar ceros
            sum_inv = sum(inverted_probs)
            normalized = [p / sum_inv for p in inverted_probs]
            return np.random.choice([0, 1, 2], p=normalized)

        return random.randint(0, 2)


class RuleBasedModel:
    """Modelo basado en reglas y heurísticas."""

    def __init__(self):
        self.predictor_markov1 = MarkovPredictor(order=1)
        self.predictor_markov2 = MarkovPredictor(order=2)
        self.predictor_freq = FrequencyPredictor()
        self.history = []
        self.predictions = []

    def fit(self, sequences: List[List[int]]):
        """Entrena los predictores internos."""
        # Entrenar Markov de orden 1
        self.predictor_markov1.fit(sequences)

        # Entrenar Markov de orden 2 (solo si hay suficientes datos)
        if all(len(seq) >= 3 for seq in sequences if len(seq) > 0):
            self.predictor_markov2.fit(sequences)

    def predict(self, recent_history: List[int]) -> int:
        """Combina múltiples predictores para hacer una predicción."""
        if not recent_history:
            return random.randint(0, 2)

        predictions = []
        confidences = []

        # 1. Markov orden 1
        if len(recent_history) >= 1:
            pattern1 = tuple(recent_history[-1:])
            pred1 = self.predictor_markov1.predict(pattern1)
            if pred1 is not None:
                predictions.append(pred1)
                # Calcular confianza basada en la frecuencia
                counter = self.predictor_markov1.transitions.get(pattern1, Counter())
                total = sum(counter.values())
                if total > 0:
                    conf = counter[pred1] / total
                    confidences.append(conf)

        # 2. Markov orden 2
        if len(recent_history) >= 2:
            pattern2 = tuple(recent_history[-2:])
            pred2 = self.predictor_markov2.predict(pattern2)
            if pred2 is not None:
                predictions.append(pred2)
                counter = self.predictor_markov2.transitions.get(pattern2, Counter())
                total = sum(counter.values())
                if total > 0:
                    conf = counter[pred2] / total
                    confidences.append(conf * 1.2)  # Más peso a patrones largos

        # 3. Predictor de frecuencia
        self.predictor_freq.history = recent_history.copy()
        self.predictor_freq.move_counts = Counter(recent_history)
        pred3 = self.predictor_freq.predict()
        predictions.append(pred3)
        confidences.append(0.5)  # Confianza media para predictor de frecuencia

        # 4. Regla simple: si el oponente está en racha, contracounter
        if len(recent_history) >= 3:
            last_three = recent_history[-3:]
            # Si repite mucho, probablemente cambie
            if len(set(last_three)) == 1:  # Todos iguales
                predictions.append((last_three[0] + 1) % 3)  # Asume que cambiará
                confidences.append(0.8)
            # Si alterna, sigue el patrón
            elif (last_three[0] != last_three[1] and
                  last_three[1] != last_three[2] and
                  last_three[0] != last_three[2]):
                # Ciclo completo, difícil predecir - usar aleatorio
                pass

        # Combinar predicciones por confianza
        if predictions:
            # Ponderar por confianza
            weighted = defaultdict(float)
            for pred, conf in zip(predictions, confidences):
                weighted[pred] += conf

            # Devolver predicción con mayor confianza
            return max(weighted.items(), key=lambda x: x[1])[0]

        # Fallback: aleatorio
        return random.randint(0, 2)


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga los datos del CSV de partidas."""
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    try:
        df = pd.read_csv(ruta_csv)

        # Verificar columnas necesarias
        required_cols = ['numero_ronda', 'jugada_j1', 'jugada_j2']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")

        print(f"Datos cargados: {len(df)} rondas")
        return df

    except FileNotFoundError:
        return crear_datos_ejemplo()
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return crear_datos_ejemplo()


def crear_datos_ejemplo() -> pd.DataFrame:
    """Crea datos de ejemplo si no hay CSV."""
    np.random.seed(42)
    n_muestras = 50

    # Patrones comunes que aparecen en jugadores humanos
    datos = []

    # Patrón 1: Repetición
    for i in range(10):
        datos.append({'numero_ronda': i + 1, 'jugada_j1': 'piedra', 'jugada_j2': 'papel'})

    # Patrón 2: Ciclo
    ciclo = ['piedra', 'papel', 'tijera']
    for i in range(15):
        j2 = ciclo[i % 3]
        j1 = ciclo[(i + 1) % 3]  # Juega para perder
        datos.append({'numero_ronda': i + 11, 'jugada_j1': j1, 'jugada_j2': j2})

    # Patrón 3: Aleatorio
    for i in range(25):
        j1 = np.random.choice(['piedra', 'papel', 'tijera'])
        j2 = np.random.choice(['piedra', 'papel', 'tijera'])
        datos.append({'numero_ronda': i + 26, 'jugada_j1': j1, 'jugada_j2': j2})

    return pd.DataFrame(datos)


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos para el modelo."""
    if df.empty:
        print("DataFrame vacío")
        return df

    df = df.copy()

    # Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(lambda x: JUGADA_A_NUM.get(str(x).lower(), 0))
    df['jugada_j2_num'] = df['jugada_j2'].map(lambda x: JUGADA_A_NUM.get(str(x).lower(), 0))

    # Crear target: próxima jugada del oponente
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Calcular resultado de cada ronda
    def calcular_resultado(row):
        j1 = row['jugada_j1_num']
        j2 = row['jugada_j2_num']

        if j1 == j2:
            return 0  # Empate
        elif (j1 == 0 and j2 == 2) or (j1 == 1 and j2 == 0) or (j1 == 2 and j2 == 1):
            return 1  # J1 gana
        else:
            return -1  # J2 gana

    df['resultado'] = df.apply(calcular_resultado, axis=1)

    # Eliminar NaN
    df = df.dropna()

    return df


# =============================================================================
# PARTE 2: ENTRENAMIENTO DEL MODELO PROPIO
# =============================================================================

def crear_secuencias_entrenamiento(df: pd.DataFrame) -> List[List[int]]:
    """Crea secuencias de jugadas para entrenar modelos de Markov."""
    sequences = []

    # Secuencia completa de jugadas del oponente
    if 'jugada_j2_num' in df.columns:
        sequences.append(df['jugada_j2_num'].astype(int).tolist())

    # También crear secuencias por ventanas
    jugadas = df['jugada_j2_num'].astype(int).tolist()
    window_size = min(20, len(jugadas))

    for i in range(0, len(jugadas) - window_size + 1, 3):
        sequences.append(jugadas[i:i + window_size])

    return sequences


def entrenar_modelo_propio(df: pd.DataFrame) -> RuleBasedModel:

    # Crear secuencias de entrenamiento
    sequences = crear_secuencias_entrenamiento(df)

    # Crear y entrenar modelo
    modelo = RuleBasedModel()
    modelo.fit(sequences)

    # Evaluar modelo en los datos
    evaluar_modelo_propio(modelo, df)

    return modelo


def evaluar_modelo_propio(modelo: RuleBasedModel, df: pd.DataFrame):
    """Evalúa el rendimiento del modelo en datos históricos."""
    if df.empty:

        return

    jugadas_j2 = df['jugada_j2_num'].astype(int).tolist()
    proximas_j2 = df['proxima_jugada_j2'].astype(int).tolist()

    correctas = 0
    total = min(len(jugadas_j2) - 3, 100)  # Evaluar máximo 100


    for i in range(3, min(len(jugadas_j2) - 1, total + 3)):
        # Usar historial reciente para predecir
        historial = jugadas_j2[max(0, i - 10):i]
        prediccion = modelo.predict(historial)

        if prediccion == proximas_j2[i]:
            correctas += 1

    if total > 0:
        accuracy = correctas / total


    # Mostrar matriz de confusión simple

    for i in range(max(0, len(jugadas_j2) - 5), len(jugadas_j2) - 1):
        if i >= 0:
            historial = jugadas_j2[max(0, i - 5):i]
            prediccion = modelo.predict(historial)
            real = proximas_j2[i]



# =============================================================================
# PARTE 3: JUGADOR IA
# =============================================================================

class JugadorIA:
    """Jugador de IA que usa nuestro modelo propio."""

    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)
        self.historial_j2 = []  # Solo jugadas del oponente (números)

        # Cargar modelo si existe
        if ruta_modelo and os.path.exists(ruta_modelo):
            try:
                with open(ruta_modelo, "rb") as f:
                    self.modelo = pickle.load(f)
                print("Modelo cargado exitosamente")
            except:

                self.modelo = RuleBasedModel()
        else:
            # Crear modelo básico
            self.modelo = RuleBasedModel()


    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """Registra una ronda jugada."""
        # Validar y normalizar jugadas
        jugada_j1 = jugada_j1.lower().strip()
        jugada_j2 = jugada_j2.lower().strip()

        if jugada_j1 not in JUGADA_A_NUM:
            jugada_j1 = "piedra"
        if jugada_j2 not in JUGADA_A_NUM:
            jugada_j2 = "piedra"

        self.historial.append((jugada_j1, jugada_j2))

        # Guardar jugada del oponente como número
        j2_num = JUGADA_A_NUM[jugada_j2]
        self.historial_j2.append(j2_num)

        # Mantener tamaño manejable
        if len(self.historial_j2) > 50:
            self.historial_j2.pop(0)
            self.historial.pop(0)

    def predecir_jugada_oponente(self) -> str:
        """Predice la próxima jugada del oponente."""
        if not self.historial_j2:
            # Sin historial, usar estrategia inicial
            return self._estrategia_inicial()

        # Usar nuestro modelo para predecir
        if self.modelo:
            try:
                prediccion_num = self.modelo.predict(self.historial_j2)
                return NUM_A_JUGADA[prediccion_num]
            except:
                pass

        # Fallback a estrategias heurísticas
        return self._estrategia_heuristica()

    def _estrategia_inicial(self) -> str:
        """Estrategia para las primeras rondas."""
        # Estadísticamente, 'papel' es ligeramente mejor al inicio
        return "papel"

    def _estrategia_heuristica(self) -> str:
        """Estrategias heurísticas cuando no hay modelo."""
        if len(self.historial_j2) < 2:
            return random.choice(["piedra", "papel", "tijera"])

        # Analizar patrones simples
        ultimas_jugadas = self.historial_j2[-5:] if len(self.historial_j2) >= 5 else self.historial_j2

        # 1. ¿Repite mucho?
        if len(set(ultimas_jugadas)) == 1:
            # Siempre juega lo mismo, asumir que cambiará
            jugada_actual = ultimas_jugadas[-1]
            siguiente = (jugada_actual + 1) % 3  # Cambia al siguiente en el ciclo
            return NUM_A_JUGADA[siguiente]

        # 2. ¿Ciclo detectado?
        if len(ultimas_jugadas) >= 3:
            # Buscar patrones cíclicos
            for ciclo in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:  # Diferentes ciclos
                if tuple(ultimas_jugadas[-3:]) == ciclo:
                    siguiente = (ciclo[-1] + 1) % 3
                    return NUM_A_JUGADA[siguiente]

        # 3. Frecuencia simple
        contador = Counter(ultimas_jugadas)
        mas_comun = contador.most_common(1)[0][0]

        # Jugar contra lo más común
        return PIERDE_CONTRA[NUM_A_JUGADA[mas_comun]]

    def decidir_jugada(self) -> str:
        """Decide qué jugada hacer para ganar."""
        prediccion_oponente = self.predecir_jugada_oponente()

        # Jugar lo que le gana a la predicción
        if prediccion_oponente in PIERDE_CONTRA:
            jugada = PIERDE_CONTRA[prediccion_oponente]
        else:
            jugada = random.choice(["piedra", "papel", "tijera"])

        # Añadir pequeña aleatoriedad para no ser predecible
        if len(self.historial) > 5 and random.random() < 0.15:
            # 15% de cambiar a una jugada diferente ocasionalmente
            opciones = [j for j in ["piedra", "papel", "tijera"] if j != jugada]
            jugada = random.choice(opciones)

        return jugada


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente guardado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():

    try:
        # 1. Cargar datos

        df = cargar_datos()

        # 2. Preparar datos

        df_prep = preparar_datos(df)

        if df_prep.empty:

            modelo_ia = JugadorIA()
        else:
            # 3. Entrenar modelo propio
            print("Entrenando modelo...")
            modelo_entrenado = entrenar_modelo_propio(df_prep)

            # Guardar modelo
            guardar_modelo(modelo_entrenado)

            # 4. Crear jugador IA con modelo entrenado
            modelo_ia = JugadorIA(RUTA_MODELO)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def entrenar_desde_consola():


    # Cargar y preparar datos
    df = cargar_datos()

    if df.empty:
        print("No se encontraron datos. Creando modelo básico.")
        modelo = RuleBasedModel()
        guardar_modelo(modelo)
        return modelo

    df_prep = preparar_datos(df)

    if len(df_prep) < 10:
        print(f"Solo {len(df_prep)} rondas válidas. Modelo básico.")
        modelo = RuleBasedModel()
    else:
        print(f"Entrenando con {len(df_prep)} rondas...")
        modelo = entrenar_modelo_propio(df_prep)

    guardar_modelo(modelo)
    print("Modelo entrenado y guardado.")

    return modelo


if __name__ == "__main__":
    main()