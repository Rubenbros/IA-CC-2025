"""
Modelo RPS JUAN JORGE Y DAVID 56% UVA PURPLE TESLA
"""

import os
import pickle
import warnings
from pathlib import Path
from collections import defaultdict, Counter, deque
import random

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_PROYECTO / "datos.csv"
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
CSV_MAP = {1: "piedra", 2: "papel", 3: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None: ruta_csv = RUTA_DATOS
    if not os.path.exists(ruta_csv):
        # Dummy data para evitar crash si no hay csv
        return pd.DataFrame({"jugada_j1": [], "jugada_j2": []})

    df = pd.read_csv(ruta_csv)
    df = df.rename(columns={
        "Partida": "numero_ronda",
        "EleccionPersona": "jugada_j1",
        "EleccionJugador2": "jugada_j2",
    })
    df["jugada_j1"] = df["jugada_j1"].map(CSV_MAP)
    df["jugada_j2"] = df["jugada_j2"].map(CSV_MAP)
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 5: return df
    df = df.copy()
    df["j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)
    df["proxima_jugada_humano"] = df["j1_num"].shift(-1)

    # Feature: Resultado de la ronda
    def check_win(row):
        h = NUM_A_JUGADA[row["j1_num"]]
        ia = NUM_A_JUGADA[row["j2_num"]]
        if GANA_A[h] == ia: return 1  # Gana Humano
        if h == ia: return 0  # Empate
        return -1  # Pierde Humano

    df["resultado"] = df.apply(check_win, axis=1)
    return df.dropna()


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 5: return df
    df = df.copy()
    # Features Lags (Profundidad media)
    df["lag_1_h"] = df["j1_num"].shift(1)
    df["lag_2_h"] = df["j1_num"].shift(2)
    df["lag_1_ia"] = df["j2_num"].shift(1)
    df["lag_1_res"] = df["resultado"].shift(1)
    return df.dropna()


def seleccionar_features(df: pd.DataFrame) -> tuple:
    cols = ["lag_1_h", "lag_2_h", "lag_1_ia", "lag_1_res"]
    if len(df) > 0 and set(cols).issubset(df.columns):
        return df[cols], df["proxima_jugada_humano"]
    return pd.DataFrame(), pd.Series()


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(X, y):
    if len(X) < 10: return None
    # Usamos todos los datos para entrenar (sin split) para máxima info histórica
    # en este punto final.
    modelo = RandomForestClassifier(n_estimators=600, max_depth=12, random_state=42)
    modelo.fit(X, y)
    return modelo


def guardar_modelo(modelo):
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)


def cargar_modelo():
    if os.path.exists(RUTA_MODELO):
        with open(RUTA_MODELO, "rb") as f:
            return pickle.load(f)
    return None


# =============================================================================
# CEREBRO
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.modelo = cargar_modelo()
        self.historial = []

        # PUNTUACIONES DE ESTRATEGIAS
        self.scores = {
            "ML": 10.0,
            "Pattern_Hum": 15.0,  # Prioridad alta inicial
            "Pattern_Full": 12.0,
            "Freq_Recent": 5.0,
            "Meta_Counter": 5.0
        }

        # Rachas de aciertos por estrategia
        self.rachas_aciertos = {k: 0 for k in self.scores.keys()}

        self.last_predictions = {}

        # MEMORIA PROFUNDA (Hasta 6)
        # Diccionario de diccionarios de contadores
        self.patrones_hum = {i: defaultdict(Counter) for i in range(1, 7)}
        self.patrones_full = {i: defaultdict(Counter) for i in range(1, 7)}

        # Memoria reciente (últimos 10 turnos)
        self.memoria_reciente = deque(maxlen=10)

        self.racha_derrotas = 0
        self.modo_inverso = False

    def registrar_ronda(self, jugada_humano: str, jugada_ia: str):
        # 0. CONTROL DE DAÑOS (MODO INVERSO)
        if GANA_A[jugada_humano] == jugada_ia:  # IA pierde
            self.racha_derrotas += 1
        else:  # IA gana o empata
            self.racha_derrotas = 0

        # Si perdemos 3 seguidas, invertir lógica 2 turnos
        if self.racha_derrotas >= 3:
            self.modo_inverso = True
        elif self.racha_derrotas == 0:
            self.modo_inverso = False

        # 1. EVALUAR ESTRATEGIAS
        if self.last_predictions:
            for nombre, prediccion in self.last_predictions.items():
                if prediccion == jugada_humano:
                    # ACIERTO
                    self.rachas_aciertos[nombre] += 1
                    # Bonus por racha: si llevas 3 aciertos, sumas mucho más
                    bonus = 2.0 + (1.0 if self.rachas_aciertos[nombre] >= 3 else 0)
                    self.scores[nombre] += bonus
                else:
                    # FALLO
                    self.rachas_aciertos[nombre] = 0
                    self.scores[nombre] *= 0.65  # Castigo severo
                    if self.scores[nombre] < 1.0: self.scores[nombre] = 1.0

        # 2. APRENDIZAJE
        self.memoria_reciente.append(jugada_humano)
        self.historial.append((jugada_humano, jugada_ia))

        target = jugada_humano
        h_moves = [h for h, _ in self.historial]
        full_moves = self.historial

        # Entrenar patrones (Profundidad 1 a 6)
        if len(self.historial) >= 2:
            for depth in range(1, 7):
                if len(self.historial) >= depth + 1:
                    # Patrón Humano
                    ctx_h = tuple(h_moves[-(depth + 1):-1])
                    self.patrones_hum[depth][ctx_h][target] += 1

                    # Patrón Completo
                    ctx_f = tuple(full_moves[-(depth + 1):-1])
                    self.patrones_full[depth][ctx_f][target] += 1

    # --- ESTRATEGIAS ---

    def predecir_ml(self):
        if self.modelo is None or len(self.historial) < 2: return None
        try:
            prev = self.historial[-2]
            curr = self.historial[-1]

            # Resultado previo
            h_prev = NUM_A_JUGADA[JUGADA_A_NUM[prev[0]]]
            ia_prev = NUM_A_JUGADA[JUGADA_A_NUM[prev[1]]]
            if GANA_A[h_prev] == ia_prev:
                res = 1
            elif h_prev == ia_prev:
                res = 0
            else:
                res = -1

            feat = np.array([[
                JUGADA_A_NUM[curr[0]],
                JUGADA_A_NUM[prev[0]],
                JUGADA_A_NUM[curr[1]],
                res
            ]])
            pred_num = self.modelo.predict(feat)[0]
            return NUM_A_JUGADA[pred_num]
        except:
            return None

    def predecir_pattern_hum(self):
        h_moves = [h for h, _ in self.historial]
        # De más largo a más corto
        for depth in [6, 5, 4, 3, 2, 1]:
            if len(h_moves) >= depth:
                ctx = tuple(h_moves[-depth:])
                if ctx in self.patrones_hum[depth]:
                    # SNIPER CHECK:
                    # Si este patrón ha ocurrido más de 2 veces y tiene una opción dominante
                    top = self.patrones_hum[depth][ctx].most_common(1)[0]
                    total_occurrences = sum(self.patrones_hum[depth][ctx].values())

                    # Si es un patrón largo (4+) y muy claro, devolver con etiqueta "SNIPER"
                    if depth >= 4 and top[1] / total_occurrences > 0.8:
                        return top[0], True  # True indica "Alta confianza"

                    if top[1] >= 1: return top[0], False
        return None, False

    def predecir_pattern_full(self):
        for depth in [6, 5, 4, 3, 2, 1]:
            if len(self.historial) >= depth:
                ctx = tuple(self.historial[-depth:])
                if ctx in self.patrones_full[depth]:
                    top = self.patrones_full[depth][ctx].most_common(1)[0]
                    total = sum(self.patrones_full[depth][ctx].values())

                    if depth >= 4 and top[1] / total > 0.8:
                        return top[0], True

                    if top[1] >= 1: return top[0], False
        return None, False

    def predecir_freq_recent(self):
        if not self.memoria_reciente: return None
        return Counter(self.memoria_reciente).most_common(1)[0][0]

    def decidir_jugada(self) -> str:
        # 1. Obtener predicciones (y flags de sniper)
        p_hum, sniper_h = self.predecir_pattern_hum() or (None, False)
        p_full, sniper_f = self.predecir_pattern_full() or (None, False)

        # --- LÓGICA SNIPER (BYPASS) ---
        # Si un patrón largo y seguro detecta algo, IGNORAR DEMOCRACIA
        if sniper_h: return PIERDE_CONTRA[p_hum]
        if sniper_f: return PIERDE_CONTRA[p_full]

        p_ml = self.predecir_ml()
        p_freq = self.predecir_freq_recent()

        # Meta-Counter
        p_meta = None
        if p_full:
            # Asumimos que el humano intentará ganar a lo que yo jugaría contra p_full
            mi_jugada_base = PIERDE_CONTRA[p_full]
            p_meta = PIERDE_CONTRA[mi_jugada_base]

        # 2. Votación Ponderada
        candidates = {
            "ML": p_ml,
            "Pattern_Hum": p_hum,
            "Pattern_Full": p_full,
            "Freq_Recent": p_freq,
            "Meta_Counter": p_meta
        }

        validas = {k: v for k, v in candidates.items() if v is not None}
        self.last_predictions = validas

        if not validas: return random.choice(["piedra", "papel", "tijera"])

        # Elegir mejor estrategia
        # Random noise muy bajo para estabilidad
        best_strat = max(validas, key=lambda k: self.scores[k] + random.uniform(0, 0.05))
        prediccion_humano = validas[best_strat]

        # 3. MODO INVERSO
        # Si la IA está en racha de derrotas, invierte su propia predicción
        mi_jugada = PIERDE_CONTRA[prediccion_humano]

        if self.modo_inverso:
            # Jugar lo que predigo que sacará el humano (empate o ganarle a su counter)
            # Simplificación efectiva: si creo que sacas Piedra, y estoy perdiendo mucho,
            # es porque tú crees que yo sacaré Papel y tú sacas Tijera.
            # Yo debería sacar Piedra.
            return prediccion_humano

        return mi_jugada


# =============================================================================
# MAIN
# =============================================================================
def main():
    try:
        df = cargar_datos()
        if len(df) > 5:
            df = preparar_datos(df)
            df_feat = crear_features(df)
            X, y = seleccionar_features(df_feat)
            modelo = entrenar_modelo(X, y)
            guardar_modelo(modelo)
            print("[OK] Modelo RPS JUAN JORGE Y DAVID 56% UVA PURPLE TESLA LISTO.")
        else:
            with open(RUTA_MODELO, "wb") as f:
                pickle.dump(None, f)
            print("[OK] Modo Zero-Data activado.")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()