"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera (Versión prueba con nombres sencillos)
Funciona con datasets pequeños para entrenar y probar la IA inmediatamente.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# ----------------------------
# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
# ----------------------------
def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    df = pd.read_csv(ruta_csv)

    df["jugada_j1"] = df["jugada_j1"].astype(str).str.strip().str.lower()
    df["jugada_j2"] = df["jugada_j2"].astype(str).str.strip().str.lower()
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    df["proxima_jugada_j2"] = df["jugada_j2_num"]

    df = df.dropna(subset=["jugada_j1_num", "jugada_j2_num", "proxima_jugada_j2"]).reset_index(drop=True)
    return df

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["lag1"] = df["jugada_j2_num"].shift(1).fillna(-1).astype(int)
    df["lag2"] = df["jugada_j2_num"].shift(2).fillna(-1).astype(int)
    df["lag3"] = df["jugada_j2_num"].shift(3).fillna(-1).astype(int)

    df["f_piedra"] = (df["jugada_j2_num"] == JUGADA_A_NUM["piedra"]).astype(int).expanding().mean()
    df["f_papel"] = (df["jugada_j2_num"] == JUGADA_A_NUM["papel"]).astype(int).expanding().mean()
    df["f_tijera"] = (df["jugada_j2_num"] == JUGADA_A_NUM["tijera"]).astype(int).expanding().mean()

    df["res_prev"] = 0
    df["racha"] = 0
    df["cambio"] = 0
    df["gano"] = 0

    df = df.fillna(0)
    return df

def seleccionar_features(df: pd.DataFrame):
    feature_cols = ["f_piedra", "f_papel", "f_tijera",
                    "lag1", "lag2", "lag3",
                    "res_prev", "racha", "cambio", "gano"]
    X = df[feature_cols].copy()
    y = df["proxima_jugada_j2"].copy().astype(int)
    return X, y

# ----------------------------
# PARTE 2: ENTRENAMIENTO
# ----------------------------
def entrenar_modelo(X, y):
    print(f"[INFO] Filas disponibles para entrenar: {len(X)}")
    if len(X) < 5:
        print(f"[ADVERTENCIA] Muy pocos datos ({len(X)} filas). Se entrenará un modelo básico para pruebas.")

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42)
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        try:
            modelo.fit(X, y)
            resultados[nombre] = modelo
            print(f"[INFO] Modelo entrenado: {nombre}")
        except Exception as e:
            print(f"[ERROR] No se pudo entrenar {nombre}: {e}")

    # Seleccionar el primer modelo entrenado
    mejor_modelo = next(iter(resultados.values()))
    return mejor_modelo

def guardar_modelo(modelo):
    os.makedirs(RUTA_MODELO.parent, exist_ok=True)
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    print(f"[INFO] Modelo guardado en {RUTA_MODELO}")

# ----------------------------
# PARTE 3: CLASE IA
# ----------------------------
class JugadorIA:
    def __init__(self):
        self.modelo = None
        try:
            with open(RUTA_MODELO, "rb") as f:
                self.modelo = pickle.load(f)
            print(f"[INFO] Modelo cargado desde {RUTA_MODELO}")
        except FileNotFoundError:
            print("[ADVERTENCIA] Modelo no encontrado. Se jugará aleatorio.")
            self.modelo = None
        self.historial = []

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        self.historial.append((jugada_j1, jugada_j2))

    def decidir_jugada(self) -> str:
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])
        # Para prueba, seguimos jugando aleatorio
        return np.random.choice(["piedra", "papel", "tijera"])

# ----------------------------
# PARTE 4: FUNCION PRINCIPAL
# ----------------------------
def main():
    print("=== RPSAI - Entrenamiento (Versión prueba) ===")
    try:
        df = cargar_datos()
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los datos: {e}")
        return
    df_feat = crear_features(df)
    X, y = seleccionar_features(df_feat)
    modelo = entrenar_modelo(X, y)
    guardar_modelo(modelo)
    print("[INFO] Entrenamiento finalizado.")

if __name__ == "__main__":
    main()