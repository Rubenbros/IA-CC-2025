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
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
    """
    Carga los datos del CSV de partidas.

    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas obligatorias.
    """

    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    ruta_csv = Path(ruta_csv)

    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_csv}")

    columnas_obligatorias = {"numero_ronda", "jugada_j1", "jugada_j2"}

    df = pd.read_csv(ruta_csv)

    faltantes = columnas_obligatorias - set(df.columns)

    if faltantes:
        raise ValueError(
            f"El CSV '{ruta_csv}' no contiene las columnas obligatorias: {faltantes}"
        )

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    # Asegurarse de estar ordenado por sesion y numero de ronda
    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
    else:
        df = df.sort_values("numero_ronda")

    # Mapear jugadas a numeros
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    # Crear la columna de próxima jugada del oponente (jugada_j2 de la siguiente ronda)
    if "session_id" in df.columns:
        df["proxima_jugada_j2_num"] = (
            df.groupby("session_id")["jugada_j2_num"].shift(-1)
        )
    else:
        df["proxima_jugada_j2_num"] = df["jugada_j2_num"].shift(-1)

    # Eliminar filas con NaN (por ejemplo, la última ronda de cada sesión)
    df = df.dropna(subset=["jugada_j1_num", "jugada_j2_num", "proxima_jugada_j2_num"])

    # Asegurar tipo entero para las columnas numéricas de jugada
    df["jugada_j1_num"] = df["jugada_j1_num"].astype(int)
    df["jugada_j2_num"] = df["jugada_j2_num"].astype(int)
    df["proxima_jugada_j2_num"] = df["proxima_jugada_j2_num"].astype(int)

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------- Feature 1: frecuencias j2 ----------
    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
    else:
        df = df.sort_values("numero_ronda")
        group = [(None, df)]

    df["freq_j2_piedra"] = 0.0
    df["freq_j2_papel"] = 0.0
    df["freq_j2_tijera"] = 0.0

    for _, g in group:
        idx = g.index
        piedra_hist = (g["jugada_j2"] == "piedra").astype(int)
        papel_hist = (g["jugada_j2"] == "papel").astype(int)
        tijera_hist = (g["jugada_j2"] == "tijera").astype(int)

        piedra_cum = piedra_hist.cumsum().shift(1)
        papel_cum = papel_hist.cumsum().shift(1)
        tijera_cum = tijera_hist.cumsum().shift(1)

        rondas_previas = pd.Series(range(len(g)), index=idx)

        df.loc[idx, "freq_j2_piedra"] = piedra_cum / rondas_previas.replace(0, pd.NA)
        df.loc[idx, "freq_j2_papel"] = papel_cum / rondas_previas.replace(0, pd.NA)
        df.loc[idx, "freq_j2_tijera"] = tijera_cum / rondas_previas.replace(0, pd.NA)

    df[["freq_j2_piedra", "freq_j2_papel", "freq_j2_tijera"]] = (
        df[["freq_j2_piedra", "freq_j2_papel", "freq_j2_tijera"]].fillna(0.0)
    )

    # ---------- Feature 2: lags ----------
    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
        df["j2_lag1"] = group["jugada_j2_num"].shift(1)
        df["j2_lag2"] = group["jugada_j2_num"].shift(2)
        df["j2_lag3"] = group["jugada_j2_num"].shift(3)
        df["j1_lag1"] = group["jugada_j1_num"].shift(1)
        df["j1_lag2"] = group["jugada_j1_num"].shift(2)
        df["j1_lag3"] = group["jugada_j1_num"].shift(3)
    else:
        df = df.sort_values("numero_ronda")
        df["j2_lag1"] = df["jugada_j2_num"].shift(1)
        df["j2_lag2"] = df["jugada_j2_num"].shift(2)
        df["j2_lag3"] = df["jugada_j2_num"].shift(3)
        df["j1_lag1"] = df["jugada_j1_num"].shift(1)
        df["j1_lag2"] = df["jugada_j1_num"].shift(2)
        df["j1_lag3"] = df["jugada_j1_num"].shift(3)

    if "resultado_num" not in df.columns and "resultado" in df.columns:
        mapa_resultado = {
            "victoria": 1,
            "empate": 0,
            "derrota": -1,
        }
        df["resultado_num"] = df["resultado"].map(mapa_resultado)

    # ---------- Feature 3: resultado anterior ----------
    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
        df["resultado_anterior_num"] = group["resultado_num"].shift(1)
    else:
        df = df.sort_values("numero_ronda")
        df["resultado_anterior_num"] = df["resultado_num"].shift(1)

    # ---------- Feature 4: racha_actual ----------
    def calcular_racha(grupo: pd.DataFrame) -> pd.Series:
        racha = []
        actual = 0
        for res in grupo["resultado_num"]:
            if res == 0:
                actual = 0
            else:
                if actual == 0 or np.sign(actual) != np.sign(res):
                    actual = res
                else:
                    actual = actual + res
            racha.append(actual)
        return pd.Series(racha, index=grupo.index)

    df["racha_actual"] = 0

    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
        for _, g in group:
            racha = calcular_racha(g)
            df.loc[g.index, "racha_actual"] = racha
    else:
        df = df.sort_values("numero_ronda")
        racha = calcular_racha(df)
        df["racha_actual"] = racha

        # ---------- Feature 5: probabilidad de que j2 juegue la jugada que gana a la ultima de j1 ----------
        # Para cada ronda, miramos el historial anterior y calculamos
        # en las rondas donde j1 jugo lo mismo que en la ronda anterior,
        # que porcentaje de veces j2 jugo la jugada que gana a esa jugada de j1.

    df["prob_j2_gana_a_ultima_j1"] = 0.0

    def calcular_prob_reaccion(grupo: pd.DataFrame) -> pd.Series:
        probs = []
        # Historial acumulado hasta ronda-1
        for i in range(len(grupo)):
            if i == 0:
                probs.append(0.0)
                continue

            sub = grupo.iloc[:i]  # historial previo
            jugada_j1_ultima = grupo.iloc[i - 1]["jugada_j1"]
            jugada_que_gana = PIERDE_CONTRA[jugada_j1_ultima]  # lo que gana a j1

            # Filtramos rondas previas donde j1 jugo lo mismo que en la ultima
            mismo_j1 = sub[sub["jugada_j1"] == jugada_j1_ultima]
            total = len(mismo_j1)
            if total == 0:
                probs.append(0.0)
                continue

            cuenta_gana = (mismo_j1["jugada_j2"] == jugada_que_gana).sum()
            probs.append(cuenta_gana / total)

        return pd.Series(probs, index=grupo.index)

    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
        for _, g in group:
            probs = calcular_prob_reaccion(g)
            df.loc[g.index, "prob_j2_gana_a_ultima_j1"] = probs
    else:
        df = df.sort_values("numero_ronda")
        df["prob_j2_gana_a_ultima_j1"] = calcular_prob_reaccion(df)

        # ---------- Feature 6: fase del juego ----------
    df["fase_juego"] = 0

    if "session_id" in df.columns:
        df = df.sort_values(["session_id", "numero_ronda"])
        group = df.groupby("session_id")
        for _, g in group:
            idx = g.index
            max_ronda = g["numero_ronda"].max()
            pos = g["numero_ronda"] / max_ronda
            fase = pd.cut(
                pos,
                bins=[0.0, 1 / 3, 2 / 3, 1.0],
                labels=[0, 1, 2],
                include_lowest=True,
                right=True,
            ).astype(int)
            df.loc[idx, "fase_juego"] = fase
    else:
        df = df.sort_values("numero_ronda")
        max_ronda = df["numero_ronda"].max()
        pos = df["numero_ronda"] / max_ronda
        df["fase_juego"] = pd.cut(
            pos,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=[0, 1, 2],
            include_lowest=True,
            right=True,
        ).astype(int)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2_num'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """
    df = df.copy()

    feature_cols = [
        # Frecuencias del oponente
        "freq_j2_piedra",
        "freq_j2_papel",
        "freq_j2_tijera",
        # Lags de jugadas
        "j2_lag1",
        "j2_lag2",
        "j2_lag3",
        "j1_lag1",
        "j1_lag2",
        "j1_lag3",
        # Resultado previo
        "resultado_anterior_num",
        # Racha actual
        "racha_actual",
        # la jugada que gana a la ultima jugada de j1
        "prob_j2_gana_a_ultima_j1",
        "fase_juego"
    ]

    # Filtrar solo columnas que existan (por si alguna feature aún no la tienes)
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Eliminar filas con NaN en features o target
    cols_para_limpieza = feature_cols + ["proxima_jugada_j2_num"]
    df = df.dropna(subset=cols_para_limpieza)

    X = df[feature_cols]
    y = df["proxima_jugada_j2_num"].astype(int)

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    - Muestra metricas de evaluacion

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        El mejor modelo entrenado
    """
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,  # mantiene proporcion de clases
    )

    # Modelos candidatos
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    mejor_modelo = None
    mejor_nombre = None
    mejor_acc = -1.0

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy {nombre}: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo
            mejor_nombre = nombre

    print(f"\nMejor modelo: {mejor_nombre} con accuracy = {mejor_acc:.4f}")
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
    """
    Clase que encapsula el modelo para jugar.

    - Carga un modelo entrenado
    - Mantiene historial de la partida actual
    - Predice la proxima jugada del oponente
    - Decide que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # Carga el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")
            self.modelo = None

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def _calcular_resultado_num(self, j1: str, j2: str) -> int:
        """
        Calcula resultado_num para una ronda:
        1 victoria j1, 0 empate, -1 derrota j1.
        """
        if j1 == j2:
            return 0
        if GANA_A[j1] == j2:
            return 1
        return -1

    def _calcular_racha(self, resultados: list[int]) -> int:
        """
        Calcula la racha_actual tras la última ronda de la lista.
        """
        actual = 0
        for res in resultados:
            if res == 0:
                actual = 0
            else:
                if actual == 0 or np.sign(actual) != np.sign(res):
                    actual = res
                else:
                    actual = actual + res
        return actual

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.
        Deben ser LAS MISMAS features que se usaron para entrenar.
        Si no hay suficiente historial, se rellenan con 0.
        """
        n = len(self.historial)

        # Si no hay historial, devolvemos features neutras
        if n == 0:
            return np.array([
                0.0, 0.0, 0.0,  # frecuencias
                0, 0, 0,  # j2_lag1-3
                0, 0, 0,  # j1_lag1-3
                0,  # resultado_anterior_num
                0,  # racha_actual
                0.0,  # prob_j2_gana_a_ultima_j1
                0,  # fase_juego
            ], dtype=float)

        jugadas_j1 = [j1 for j1, _ in self.historial]
        jugadas_j2 = [j2 for _, j2 in self.historial]

        # Frecuencias j2
        count_piedra = sum(1 for j in jugadas_j2 if j == "piedra")
        count_papel = sum(1 for j in jugadas_j2 if j == "papel")
        count_tijera = sum(1 for j in jugadas_j2 if j == "tijera")

        freq_j2_piedra = count_piedra / n
        freq_j2_papel = count_papel / n
        freq_j2_tijera = count_tijera / n

        # Lags
        def lag(lista, k):
            if n - k >= 0:
                return JUGADA_A_NUM[lista[n - k]]
            return 0

        j2_lag1 = lag(jugadas_j2, 1)
        j2_lag2 = lag(jugadas_j2, 2)
        j2_lag3 = lag(jugadas_j2, 3)

        j1_lag1 = lag(jugadas_j1, 1)
        j1_lag2 = lag(jugadas_j1, 2)
        j1_lag3 = lag(jugadas_j1, 3)

        # Resultado anterior y racha
        resultados = [
            self._calcular_resultado_num(j1, j2)
            for j1, j2 in self.historial
        ]
        resultado_anterior_num = resultados[-1]
        racha_actual = self._calcular_racha(resultados)

        # prob_j2_gana_a_ultima_j1 (misma idea que en crear_features, pero usando historial completo)
        if n == 1:
            prob_j2_gana_a_ultima_j1 = 0.0
        else:
            # usamos todas las rondas menos la ultima como historial
            sub_hist = self.historial[:-1]
            ultima_j1 = self.historial[-1][0]
            jugada_que_gana = PIERDE_CONTRA[ultima_j1]
            # rondas previas donde j1 jugo igual
            mismo_j1 = [(j1, j2) for (j1, j2) in sub_hist if j1 == ultima_j1]
            total = len(mismo_j1)
            if total == 0:
                prob_j2_gana_a_ultima_j1 = 0.0
            else:
                cuenta_gana = sum(1 for (_, j2) in mismo_j1 if j2 == jugada_que_gana)
                prob_j2_gana_a_ultima_j1 = cuenta_gana / total

        # fase_juego: aproximacion simple basada en la ronda actual
        # supondremos una sesion "larga" y normalizamos n respecto a un max fijo, por ejemplo 50
        max_rondas_sesion = 50
        pos = min(n / max_rondas_sesion, 1.0)
        if pos <= 1 / 3:
            fase_juego = 0
        elif pos <= 2 / 3:
            fase_juego = 1
        else:
            fase_juego = 2

        features = np.array([
            freq_j2_piedra,
            freq_j2_papel,
            freq_j2_tijera,
            j2_lag1,
            j2_lag2,
            j2_lag3,
            j1_lag1,
            j1_lag2,
            j1_lag3,
            resultado_anterior_num,
            racha_actual,
            prob_j2_gana_a_ultima_j1,
            fase_juego,
        ], dtype=float)

        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        Usa obtener_features_actuales() y el modelo entrenado.
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales().reshape(1, -1)
        pred_num = self.modelo.predict(features)[0]
        return NUM_A_JUGADA[int(pred_num)]

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("=" * 50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    # 1. Cargar datos
    try:
        df_raw = cargar_datos()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"\n[+] Datos cargados: {len(df_raw)} filas")

    # 2. Preparar datos
    df_prep = preparar_datos(df_raw)
    print(f"[+] Datos preparados: {len(df_prep)} filas")

    # 3. Crear features
    df_feat = crear_features(df_prep)
    print(f"[+] Features creadas. Columnas totales: {len(df_feat.columns)}")

    # 4. Seleccionar features (X, y)
    X, y = seleccionar_features(df_feat)
    print(f"[+] Features seleccionadas: X.shape = {X.shape}, y.shape = {y.shape}")

    if len(X) == 0:
        print("[ERROR] No hay suficientes datos despues de crear features/filtrar NaN.")
        return

    # 5. Entrenar modelo
    print("\n[+] Entrenando modelos...")
    mejor_modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    if mejor_modelo is not None:
        guardar_modelo(mejor_modelo)
    else:
        print("[ERROR] No se entreno ningun modelo valido.")


if __name__ == "__main__":
    main()
