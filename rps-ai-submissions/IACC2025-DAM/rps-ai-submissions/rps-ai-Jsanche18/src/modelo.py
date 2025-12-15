"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera (MEJORADO + ANALISIS REAL)
============================================================================
Código adaptado para que la IA estudie el estilo del rival, analice patrones,
y no juegue aleatorio salvo cuando no confía en su predicción.
"""

import os
import pickle
import warnings
from pathlib import Path
import random

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        print("AVISO: No existe el CSV, creando datos sintéticos...")
        return pd.DataFrame({
            'numero_ronda': range(1, 201),
            'jugada_j1': np.random.choice(['piedra', 'papel', 'tijera'], 200),
            'jugada_j2': np.random.choice(['piedra', 'papel', 'tijera'], 200)
        })

    try:
        return pd.read_csv(ruta_csv)
    except:
        return pd.DataFrame()


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)
    df['proxima_jugada_j2'] = df['j2_num'].shift(-1)
    return df.dropna()


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['feat_j2_lag1'] = df['j2_num']
    df['feat_j2_lag2'] = df['j2_num'].shift(1).fillna(0)
    df['feat_j1_lag1'] = df['j1_num']
    df['feat_patron'] = (df['j2_num'].shift(1).fillna(0) * 3) + df['j2_num']
    df['feat_resultado'] = (df['j1_num'] - df['j2_num']) % 3

    return df.dropna()


def seleccionar_features(df: pd.DataFrame):
    features = [c for c in df.columns if c.startswith("feat_")]
    return df[features], df['proxima_jugada_j2']


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"Accuracy del modelo: {acc:.2%}")

    return modelo


def guardar_modelo(modelo):
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)


def cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        return None
    with open(RUTA_MODELO, "rb") as f:
        return pickle.load(f)


# ============================================================
# IA MEJORADA CON ANALISIS REAL DEL RIVAL
# ============================================================

class JugadorIA:
    def __init__(self, ruta_modelo=None):
        self.modelo = cargar_modelo()
        self.historial = []
        self.racha_perdidas = 0
        self.perfil_rival = {}

    # ----------------------------------------
    # Guardar las rondas que se van jugando
    # ----------------------------------------
    def registrar_ronda(self, jugada_j1, jugada_j2):
        self.historial.append((jugada_j1, jugada_j2))

        gana_j1 = (JUGADA_A_NUM[jugada_j1] - JUGADA_A_NUM[jugada_j2]) % 3 == 1
        empate = jugada_j1 == jugada_j2

        if not gana_j1 and not empate:
            self.racha_perdidas += 1
        else:
            self.racha_perdidas = 0

    # ----------------------------------------
    # Análisis del estilo del rival
    # ----------------------------------------
    def analizar_estilo(self):
        if len(self.historial) < 5:
            return

        j1 = [JUGADA_A_NUM[x[0]] for x in self.historial]  # humano
        j2 = [JUGADA_A_NUM[x[1]] for x in self.historial]  # IA

        # Frecuencias simples
        self.perfil_rival["freq"] = {
            "piedra": j1.count(0),
            "papel": j1.count(1),
            "tijera": j1.count(2)
        }

        # Transiciones
        trans = {(i, j): 0 for i in range(3) for j in range(3)}
        for i in range(len(j1) - 1):
            trans[(j1[i], j1[i + 1])] += 1

        mejor_trans = max(trans, key=trans.get)
        self.perfil_rival["transicion_mas_comun"] = mejor_trans

        # Qué juega después de perder
        despues_perder = []
        for i in range(len(j1) - 1):
            if (j2[i] - j1[i]) % 3 == 1:  # humano perdió
                despues_perder.append(j1[i + 1])

        if despues_perder:
            mas = max(set(despues_perder), key=despues_perder.count)
            self.perfil_rival["respuesta_perdedor"] = NUM_A_JUGADA[mas]
        else:
            self.perfil_rival["respuesta_perdedor"] = None

    # ----------------------------------------
    # Features actuales para el modelo ML
    # ----------------------------------------
    def obtener_features_actuales(self):
        if len(self.historial) < 3:
            return None

        ult = self.historial[-1]
        ant = self.historial[-2]

        j1_actual = JUGADA_A_NUM[ult[0]]
        j2_actual = JUGADA_A_NUM[ult[1]]
        j2_prev = JUGADA_A_NUM[ant[1]]

        return np.array([[
            j2_actual,
            j2_prev,
            j1_actual,
            (j2_prev * 3) + j2_actual,
            (j1_actual - j2_actual) % 3
        ]])

    # ----------------------------------------
    # Predicción ML
    # ----------------------------------------
    def predecir_jugada_oponente(self):
        if self.modelo is None:
            return None

        f = self.obtener_features_actuales()
        if f is None:
            return None

        probs = self.modelo.predict_proba(f)[0]
        conf = max(probs)
        pred = np.argmax(probs)

        if conf < 0.45:
            return None

        return NUM_A_JUGADA[pred]

    # ----------------------------------------
    # DECISIÓN FINAL DE LA IA
    # ----------------------------------------
    def decidir_jugada(self):
        # Analizar estilo en rondas clave
        if len(self.historial) in [10, 15, 20, 25]:
            self.analizar_estilo()

        # Modo caos si nos destrozan
        if self.racha_perdidas >= 2:
            print("(!) Racha negativa → Modo caos")
            return random.choice(["piedra", "papel", "tijera"])

        # 1) Intento con patrón fuerte
        if "transicion_mas_comun" in self.perfil_rival:
            ult_humano = JUGADA_A_NUM[self.historial[-1][0]]
            prev, siguiente = self.perfil_rival["transicion_mas_comun"]

            if ult_humano == prev:
                pred = NUM_A_JUGADA[siguiente]
                return PIERDE_CONTRA[pred]

        # 2) Intento ML
        pred_ml = self.predecir_jugada_oponente()
        if pred_ml is not None:
            return PIERDE_CONTRA[pred_ml]

        # 3) Aleatorio seguro
        return random.choice(["piedra", "papel", "tijera"])


# ============================================================
# EJECUCIÓN
# ============================================================

def main():
    print("--- Entrenando RPS AI Mejorada ---")

    df = cargar_datos()
    if df.empty:
        print("CSV vacío o roto.")
        return

    df = preparar_datos(df)
    df = crear_features(df)

    if len(df) < 10:
        print("Datos insuficientes para entrenar.")
        return

    X, y = seleccionar_features(df)
    modelo = entrenar_modelo(X, y)
    guardar_modelo(modelo)

    print("Modelo guardado correctamente.")


if __name__ == "__main__":
    main()
