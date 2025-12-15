"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    columnas_necesarias = ["numero_ronda", "jugada_j1", "jugada_j2"]
    for col in columnas_necesarias:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir jugadas a números
    df["j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    # Target: la próxima jugada del oponente
    df["proxima_jugada_j2"] = df["j2_num"].shift(-1)

    # Eliminar filas sin target
    df = df.dropna()

    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Feature 1: Frecuencia acumulada de cada jugada del oponente ---
    df["freq_piedra"] = (df["j2_num"] == 0).expanding().mean()
    df["freq_papel"] = (df["j2_num"] == 1).expanding().mean()
    df["freq_tijera"] = (df["j2_num"] == 2).expanding().mean()

    # --- Feature 2: Lag features (últimas jugadas del oponente) ---
    df["j2_lag1"] = df["j2_num"].shift(1)
    df["j2_lag2"] = df["j2_num"].shift(2)
    df["j2_lag3"] = df["j2_num"].shift(3)

    # --- Feature 3: Resultado anterior ---
    #  1 = ganó j1, 0 = empate, -1 = perdió j1
    def resultado(j1, j2):
        if j1 == j2:
            return 0
        if GANA_A[NUM_A_JUGADA[j1]] == NUM_A_JUGADA[j2]:
            return 1
        return -1

    df["resultado_prev"] = df.apply(
        lambda row: resultado(row["j1_num"], row["j2_num"]), axis=1
    ).shift(1)

    # Quitar las primeras filas que tendrán NaN por los lags
    df = df.dropna()

    return df

def seleccionar_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "freq_piedra", "freq_papel", "freq_tijera",
        "j2_lag1", "j2_lag2", "j2_lag3",
        "resultado_prev"
    ]

    X = df[feature_cols]
    y = df["proxima_jugada_j2"]

    return X, y

# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def entrenar_modelo(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=200)
    }

    mejor_modelo = None
    mejor_acc = -1

    print("\n----- Evaluación de Modelos -----\n")

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"Modelo: {nombre}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, pred))
        print("--------------------------------\n")

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo

    print(f"[+] Mejor modelo: {mejor_modelo.__class__.__name__} (ACC={mejor_acc:.4f})")

    return mejor_modelo

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
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================
class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.historial = []
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Modelo IA cargado correctamente")
        except:
            print("No se pudo cargar el modelo. Usaré jugadas aleatorias.")
            self.modelo = None

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self):
        if len(self.historial) == 0:
            return np.zeros(7)

        df = pd.DataFrame(self.historial, columns=["j1", "j2"])
        df["j1_num"] = df["j1"].map(JUGADA_A_NUM)
        df["j2_num"] = df["j2"].map(JUGADA_A_NUM)

        # Frecuencias
        freq = df["j2_num"].value_counts(normalize=True)
        f_p = freq.get(0, 0)
        f_pa = freq.get(1, 0)
        f_t = freq.get(2, 0)

        # Lags
        lag1 = df["j2_num"].iloc[-1]
        lag2 = df["j2_num"].iloc[-2] if len(df) > 1 else 1
        lag3 = df["j2_num"].iloc[-3] if len(df) > 2 else 1

        # Resultado previo
        def res(j1, j2):
            if j1 == j2: return 0
            if GANA_A[NUM_A_JUGADA[j1]] == NUM_A_JUGADA[j2]:
                return 1
            return -1

        res_prev = res(df["j1_num"].iloc[-1], df["j2_num"].iloc[-1])

        return np.array([f_p, f_pa, f_t, lag1, lag2, lag3, res_prev])

    def predecir_jugada_oponente(self) -> str:
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        feat = self.obtener_features_actuales().reshape(1, -1)
        pred = self.modelo.predict(feat)[0]
        return NUM_A_JUGADA[pred]

    def decidir_jugada(self) -> str:
        pred = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[pred]

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================
def main():
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    df = cargar_datos()
    df = preparar_datos(df)
    df = crear_features(df)
    X, y = seleccionar_features(df)

    modelo = entrenar_modelo(X, y)
    guardar_modelo(modelo)

    print("\nEntrenamiento completado con éxito.")

if __name__ == "__main__":
    main()
