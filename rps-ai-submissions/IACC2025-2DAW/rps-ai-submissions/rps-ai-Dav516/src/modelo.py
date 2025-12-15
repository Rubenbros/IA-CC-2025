# src/modelo.py
"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
Versión adaptada para el layout:
  PROJECT_ROOT/
    data/        <- aqui debe estar el CSV (partidas.csv o cualquier .csv)
    models/      <- aqui se guardará modelo_entrenado.pkl
    src/
      modelo.py  <- este archivo
"""
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Rutas base (robustas)
# -------------------------
RUTA_PROYECTO = Path(__file__).parent.parent.resolve()   # project root
DATA_DIR = RUTA_PROYECTO / "data"
MODELS_DIR = RUTA_PROYECTO / "models"
# nombre por defecto de salida del modelo
RUTA_MODELO = MODELS_DIR / "modelo_entrenado.pkl"

# posibles nombres de CSV a buscar (en orden)
POSSIBLE_CSV_NAMES = ["partidas.csv", "mis_partidas_ppt.csv", "mis_partidas.csv"]

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# Parámetros de feature engineering
LAGS = 3
WINDOW_FREQ = None  # expanding mean


# -------------------------
# UTIL: encontrar CSV en data/
# -------------------------
def encontrar_csv_data(data_dir: Path = None) -> Path:
    """
    Busca un CSV válido dentro de data/ y devuelve su Path.
    Revisa primero nombres conocidos, si no encuentra toma el primer .csv.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta data/ en: {data_dir}")

    # comprobar nombres preferidos
    for name in POSSIBLE_CSV_NAMES:
        p = data_dir / name
        if p.exists():
            return p

    # si no, coger el primer .csv en la carpeta
    csvs = sorted(list(data_dir.glob("*.csv")))
    if csvs:
        return csvs[0]

    raise FileNotFoundError(f"No se encontró ningún archivo .csv en {data_dir}. Añade tu CSV (ej: partidas.csv).")


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga datos desde data/*.csv. Si ruta_csv es None intentará auto-detectar.
    """
    if ruta_csv is None:
        ruta_csv = encontrar_csv_data()
    ruta_csv = Path(ruta_csv)
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    required = {"numero_ronda", "jugada_j1", "jugada_j2"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"El CSV debe contener las columnas: {required}. Columnas encontradas: {list(df.columns)}")
    df = df.sort_values("numero_ronda").reset_index(drop=True)
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jugada_j1"] = df["jugada_j1"].astype(str).str.strip().str.lower()
    df["jugada_j2"] = df["jugada_j2"].astype(str).str.strip().str.lower()
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)
    df["proxima_jugada_j2"] = df["jugada_j2_num"].shift(-1)
    df = df.dropna(subset=["jugada_j1_num", "jugada_j2_num", "proxima_jugada_j2"])
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)
    df["jugada_j1_num"] = df["jugada_j1_num"].astype(int)
    df["jugada_j2_num"] = df["jugada_j2_num"].astype(int)
    df = df.reset_index(drop=True)
    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # frecuencias acumuladas
    for move, code in JUGADA_A_NUM.items():
        col_name = f"freq_j2_{move}"
        if WINDOW_FREQ is None:
            df[col_name] = (df["jugada_j2_num"] == code).expanding().mean()
        else:
            df[col_name] = (df["jugada_j2_num"] == code).rolling(window=WINDOW_FREQ, min_periods=1).mean()
    # lags
    for lag in range(1, LAGS + 1):
        df[f"j2_lag_{lag}"] = df["jugada_j2_num"].shift(lag)
    # resultado anterior (desde j1)
    def resultado_j1(row):
        j1 = row["jugada_j1"]
        j2 = row["jugada_j2"]
        if j1 == j2:
            return 0
        if GANA_A.get(j1) == j2:
            return 1
        return -1
    df["resultado_actual"] = df.apply(resultado_j1, axis=1)
    df["resultado_anterior"] = df["resultado_actual"].shift(1).fillna(0).astype(int)
    # j2_result y racha
    def j2_gano(row):
        j1 = row["jugada_j1"]
        j2 = row["jugada_j2"]
        if j1 == j2:
            return 0
        if GANA_A.get(j2) == j1:
            return 1
        return -1
    df["j2_result"] = df.apply(j2_gano, axis=1)
    racha = []
    current_sign = None
    count = 0
    for val in df["j2_result"].tolist():
        if current_sign is None or np.sign(val) != current_sign:
            current_sign = np.sign(val)
            count = 1
        else:
            count += 1
        racha.append(count)
    df["j2_racha"] = racha
    # ronda normalizada
    max_round = df["numero_ronda"].max() if "numero_ronda" in df.columns else len(df)
    df["ronda_norm"] = df["numero_ronda"] / max(1, max_round)
    # lags fill
    for lag in range(1, LAGS + 1):
        df[f"j2_lag_{lag}"] = df[f"j2_lag_{lag}"].fillna(-1).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


def seleccionar_features(df: pd.DataFrame):
    feature_cols = []
    for move in JUGADA_A_NUM.keys():
        feature_cols.append(f"freq_j2_{move}")
    for lag in range(1, LAGS + 1):
        feature_cols.append(f"j2_lag_{lag}")
    feature_cols += ["resultado_anterior", "j2_racha", "ronda_norm"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing}")
    X = df[feature_cols].copy()
    y = df["proxima_jugada_j2"].copy()
    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO / SELECCION
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    resultados = {}
    for name, model in modelos.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        resultados[name] = {"model": model, "accuracy": acc, "report": classification_report(y_test, y_pred, output_dict=True), "confusion": confusion_matrix(y_test, y_pred)}
        print(f"\nModelo: {name}  Accuracy: {acc:.4f}")
    best_name = max(resultados.keys(), key=lambda n: resultados[n]["accuracy"])
    best_model = resultados[best_name]["model"]
    print(f"\nMejor modelo: {best_name} con accuracy {resultados[best_name]['accuracy']:.4f}")
    return best_model, resultados


# =============================================================================
# GUARDAR / CARGAR MODELO (asegura que models/ exista)
# =============================================================================

def guardar_modelo(modelo, ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# CLASE JugadorIA (usa el mismo esquema de features)
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.historial = []
        self.feature_cols = []
        # columnas esperadas
        for move in JUGADA_A_NUM.keys():
            self.feature_cols.append(f"freq_j2_{move}")
        for lag in range(1, LAGS + 1):
            self.feature_cols.append(f"j2_lag_{lag}")
        self.feature_cols += ["resultado_anterior", "j2_racha", "ronda_norm"]
        # cargar modelo si existe
        if ruta_modelo is None:
            ruta_modelo = RUTA_MODELO
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Modelo cargado.")
        except FileNotFoundError:
            print("No se pudo cargar el modelo (entrena primero).")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        self.historial.append((jugada_j1.strip().lower(), jugada_j2.strip().lower()))

    def obtener_features_actuales(self) -> np.ndarray:
        if len(self.historial) == 0:
            sample = np.zeros(len(self.feature_cols), dtype=float)
            idx = len(JUGADA_A_NUM)  # skip freq zeros
            for lag in range(1, LAGS + 1):
                sample[idx] = -1
                idx += 1
            sample[idx] = 0; idx += 1
            sample[idx] = 0; idx += 1
            sample[idx] = 0.0
            return sample
        temp = pd.DataFrame(self.historial, columns=["jugada_j1", "jugada_j2"])
        temp["numero_ronda"] = np.arange(1, len(temp)+1)
        temp["jugada_j2_num"] = temp["jugada_j2"].map(JUGADA_A_NUM)
        feats = {}
        for move, code in JUGADA_A_NUM.items():
            if WINDOW_FREQ is None:
                vals = (temp["jugada_j2_num"] == code).expanding().mean()
            else:
                vals = (temp["jugada_j2_num"] == code).rolling(window=WINDOW_FREQ, min_periods=1).mean()
            feats[f"freq_j2_{move}"] = vals.iloc[-1]
        for lag in range(1, LAGS + 1):
            val = temp["jugada_j2_num"].shift(lag).iloc[-1] if len(temp) - lag - 0 >= 0 else -1
            feats[f"j2_lag_{lag}"] = -1 if pd.isna(val) else int(val)
        def resultado_j1_row(j1, j2):
            if j1 == j2:
                return 0
            if GANA_A.get(j1) == j2:
                return 1
            return -1
        if len(temp) >= 2:
            prev_row = temp.iloc[-2]
            res_ant = resultado_j1_row(prev_row["jugada_j1"], prev_row["jugada_j2"])
        else:
            res_ant = 0
        feats["resultado_anterior"] = int(res_ant)
        # racha simple
        racha = 0
        sign_prev = None
        for _, row in temp.iterrows():
            val = row["jugada_j2"]
            j1 = row["jugada_j1"]
            if j1 == val:
                sign = 0
            elif GANA_A.get(val) == j1:
                sign = 1
            else:
                sign = -1
            if sign_prev is None or sign != sign_prev:
                count = 1
            else:
                count += 1
            sign_prev = sign
            racha = count
        feats["j2_racha"] = int(racha)
        feats["ronda_norm"] = temp["numero_ronda"].max() / max(1, temp["numero_ronda"].max())
        feature_vector = [feats.get(c, 0) for c in self.feature_cols]
        return np.array(feature_vector, dtype=float)

    import pandas as pd
    import numpy as np

    def predecir_jugada_oponente(self) -> str:
        if self.modelo is None:
            return np.random.choice(list(JUGADA_A_NUM.keys()))

        # obtener features como array (1, n_features)
        features = self.obtener_features_actuales().reshape(1, -1)

        # Comprobar dimensión y convertir a DataFrame con nombres de columna
        if features.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Dimensión features ({features.shape[1]}) no coincide con feature_cols ({len(self.feature_cols)})."
            )

        df_features = pd.DataFrame(features, columns=self.feature_cols)

        try:
            pred = self.modelo.predict(df_features)[0]
            return NUM_A_JUGADA.get(int(pred), np.random.choice(list(JUGADA_A_NUM.keys())))
        except Exception as e:
            print("Error prediccion:", e)
            return np.random.choice(list(JUGADA_A_NUM.keys()))

    def decidir_jugada(self) -> str:
        pred = self.predecir_jugada_oponente()
        return PIERDE_CONTRA.get(pred, np.random.choice(list(JUGADA_A_NUM.keys())))


# =============================================================================
# MAIN: flujo de entrenamiento
# =============================================================================

def main():
    print("="*60)
    print("RPSAI - Entrenamiento del Modelo")
    print("="*60)
    try:
        df_raw = cargar_datos()
    except Exception as e:
        print("Error cargando datos:", e)
        return
    df_prepared = preparar_datos(df_raw)
    print(f"Datos preparados: {len(df_prepared)} filas")
    df_feat = crear_features(df_prepared)
    print(f"Features: {df_feat.shape[1]} cols  {df_feat.shape[0]} filas")
    X, y = seleccionar_features(df_feat)
    best_model, resultados = entrenar_modelo(X, y)
    # asegurar carpeta models
    os.makedirs(MODELS_DIR, exist_ok=True)
    guardar_modelo(best_model, ruta=RUTA_MODELO)
    print("\nEntrenamiento completado. Modelos evaluados:")
    for name, info in resultados.items():
        print(f" - {name}: accuracy={info['accuracy']:.4f}")

if __name__ == "__main__":
    main()
