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

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

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
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    # Pista: usa pd.read_csv()
        # Cargar CSV con pandas
        try:
            df = pd.read_csv(ruta_csv)
        except Exception as e:
            raise ValueError(f"No se pudo leer el CSV: {e}")

        # Columnas necesarias para entrenar el modelo
        columnas_requeridas = {
            "numero_ronda",
            "jugada_j1",
            "jugada_j2",
            "resultado",
            "cambia_j1",
            "cambia_j2"
        }

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
    # Pistas:
    # - Usa map() con JUGADA_A_NUM para convertir jugadas a numeros
    # - Usa shift(-1) para crear la columna de proxima jugada
    # - Usa dropna() para eliminar filas con NaN

    # Convertir jugadas de texto a números
    df = df.copy()
    df["jugada_j1"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2"] = df["jugada_j2"].map(JUGADA_A_NUM)

    #Convertir el cambia o no
    df["cambia_j1"] = df["cambia_j1"].astype(int)
    df["cambia_j2"] = df["cambia_j2"].astype(int)

    # Crear la columna target: la jugada futura de j2
    df["proxima_jugada_j2"] = df["jugada_j2"].shift(-1)

    # Eliminar filas con valores nulos (por último shift)
    df = df.dropna()

    # Convertir el target a int  dropna es un float
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    print("Después de preparar_datos:", df.shape)

    return df

# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.
    Ideas de features:
    1. Frecuencia de cada jugada del oponente (j2)
    2. Ultimas N jugadas (lag features)
    3. Resultado de la ronda anterior
    4. Racha actual (victorias/derrotas consecutivas)
    5. Patron despues de ganar/perder
    6. Fase del juego (inicio/medio/final)

    Cuantas mas features relevantes crees, mejor podra predecir tu modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # ------------------------------------------
    # Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()

    for jugada in [0, 1, 2]:  # piedra papel tijera
        df[f"freq_j2_{jugada}"] = (df["jugada_j2"] == jugada).expanding().mean()

    # ------------------------------------------
    # Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.

        df["j2_lag1"] = df["jugada_j2"].shift(1).fillna(0)
        df["j2_lag2"] = df["jugada_j2"].shift(2).fillna(0)
        df["j2_lag3"] = df["jugada_j2"].shift(3).fillna(0)

    # ------------------------------------------
    # Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    mapeo_resultado = {"j1": 1, "Empate": 0, "j2": -1}
    df["resultado_cod"] = df["resultado"].map(mapeo_resultado)

    df["resultado_anterior"] = df["resultado_cod"].shift(1).fillna(0)

    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion

    # ------------------------------------------------------
    # FEATURE 4 - Racha actual
    # ------------------------------------------------------
    # +1 si j1 gana, -1 si pierde, 0 si empata
    df["racha"] = df["resultado_cod"].groupby((df["resultado_cod"] != df["resultado_cod"].shift()).cumsum()).cumsum()

    # ------------------------------------------------------
    # FEATURE 5 - Fase del juego
    # ------------------------------------------------------
    # Se asigna una fase según el porcentaje del progreso en la partida si es al comienzo de la partida 0, mitad 0.5 final 1
    n = len(df)
    df["fase"] = (df.index / n)

    # ------------------------------------------------------
    # FEATURE 6 - Cambia despues de ganar
    # ------------------------------------------------------
    df["cambia_despues_ganar"] = ((df["resultado_cod"].shift(1) == -1) &
                                  (df["cambia_j2"] == 1)).astype(int)

    df["cambia_despues_perder"] = ((df["resultado_cod"].shift(1) == 1) &
                                   (df["cambia_j2"] == 1)).astype(int)

    # ------------------------------------------------------
    # FEATURE 7 - Cambia respecto a la jugada anterior de j1
    # ------------------------------------------------------
    df["j2_contra_j1"] = ((df["jugada_j1"].shift(1) + 1) % 3 == df["jugada_j2"]).astype(int)

    # ------------------------------------------
    # Eliminar filas con NaN generadas por lags y shift
    # ------------------------------------------
    df = df.dropna(subset=["proxima_jugada_j2"])

    print("Después de crear_features:", df.shape)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.


    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """

    # Columnas a excluir del modelo (target + identificadores)
    excluir = ["numero_ronda", "proxima_jugada_j2", "resultado"]


    # feature_cols = ['feature1', 'feature2', ...]
    feature_cols = [col for col in df.columns if col not in excluir]


    # X = df[feature_cols]
    # y = df['proxima_jugada_j2']

    X = df[feature_cols]
    y = df["proxima_jugada_j2"]

    X = X.fillna(0)
    y = y.loc[X.index]

    print("X.shape, y.shape:", X.shape, y.shape)

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
    # Divide los datos
    # X_train, X_test, y_train, y_test = train_test_split(...)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Entrena varios modelos
    # modelos = {
    #     'KNN': KNeighborsClassifier(n_neighbors=5),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'RandomForest': RandomForestClassifier()
    # }
    # Entrenamos dos modelos complementarios:
    # KNN: sirve como baseline simple, fácil de entender, funciona bien con pocos datos.
    # Random Forest: más potente, captura patrones complejos entre las features (lags, cambios, respuesta a j1), es robusto y generalmente ofrece mejor accuracy en este tipo de problemas.
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    mejores_metricas = 0
    mejor_modelo = None

    # Evalua cada modelo
    # Para cada modelo:
    #   - Entrena con fit()
    #   - Predice con predict()
    #   - Calcula accuracy con accuracy_score()
    #   - Muestra classification_report()

    # Selecciona y retorna el mejor modelo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nModelo: {nombre}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Seleccionar el mejor modelo basado en accuracy
        if acc > mejores_metricas:
            mejores_metricas = acc
            mejor_modelo = modelo

    print(f"\n El mejor modelo es : {mejor_modelo.__class__.__name__} con un acierto de:  {mejores_metricas:.4f}")

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
        return joblib.load(ruta)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

     Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        #Carga el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")




    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con 1 fila y las mismas columnas que se usaron al entrenar.
        """
        # Lista de columnas exactas usadas en entrenamiento
        feature_cols = ['freq_j2_0', 'freq_j2_1', 'freq_j2_2',
                        'j2_lag1', 'j2_lag2', 'j2_lag3',
                        'resultado_anterior', 'racha', 'fase',
                        'cambia_despues_ganar', 'cambia_despues_perder',
                        'j2_contra_j1', 'cambia_j1', 'cambia_j2',
                        'extra1', 'extra2', 'extra3']  # ajustar si tienes más o menos

        if not self.historial:
            # Devuelve ceros si no hay historial
            return pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

        df = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2'])
        df["jugada_j1"] = df["jugada_j1"].map(JUGADA_A_NUM)
        df["jugada_j2"] = df["jugada_j2"].map(JUGADA_A_NUM)
        df["cambia_j2"] = df["jugada_j2"].diff().fillna(0).ne(0).astype(int)

        # Calcular features
        features = {}
        features['freq_j2_0'] = (df['jugada_j2'] == 0).mean()
        features['freq_j2_1'] = (df['jugada_j2'] == 1).mean()
        features['freq_j2_2'] = (df['jugada_j2'] == 2).mean()
        features['j2_lag1'] = df['jugada_j2'].iloc[-1] if len(df) >= 1 else 0
        features['j2_lag2'] = df['jugada_j2'].iloc[-2] if len(df) >= 2 else 0
        features['j2_lag3'] = df['jugada_j2'].iloc[-3] if len(df) >= 3 else 0

        # Resultado anterior
        ultimo_j1, ultimo_j2 = df['jugada_j1'].iloc[-1], df['jugada_j2'].iloc[-1]
        if ultimo_j1 == ultimo_j2:
            features['resultado_anterior'] = 0
        elif (ultimo_j1 + 1) % 3 == ultimo_j2:
            features['resultado_anterior'] = -1
        else:
            features['resultado_anterior'] = 1

        # Racha
        racha = 0
        for i in range(len(df)):
            j1, j2 = df['jugada_j1'].iloc[i], df['jugada_j2'].iloc[i]
            if j1 == j2:
                val = 0
            elif (j1 + 1) % 3 == j2:
                val = -1
            else:
                val = 1
            racha += val
        features['racha'] = racha

        # Fase del juego
        features['fase'] = len(df) / 100

        # Cambia después de ganar/perder
        features['cambia_despues_ganar'] = int(
            (features['resultado_anterior'] == -1) and (df['cambia_j2'].iloc[-1] == 1))
        features['cambia_despues_perder'] = int(
            (features['resultado_anterior'] == 1) and (df['cambia_j2'].iloc[-1] == 1))

        # J2 contra j1 anterior
        if len(df) >= 2:
            features['j2_contra_j1'] = int((df['jugada_j1'].iloc[-2] + 1) % 3 == df['jugada_j2'].iloc[-1])
        else:
            features['j2_contra_j1'] = 0

        # Completar las 3 columnas extras con ceros si no se usan
        features['cambia_j1'] = 0
        features['cambia_j2'] = df['cambia_j2'].iloc[-1] if len(df) >= 1 else 0
        features['extra1'] = 0
        features['extra2'] = 0
        features['extra3'] = 0

        # Convertir a DataFrame 1 fila
        return pd.DataFrame([features], columns=feature_cols)

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

        features = self.obtener_features_actuales()
        # Convertir a numpy array 2D si es necesario
        if isinstance(features, pd.DataFrame):
            features_array = features.to_numpy()  # DataFrame a array
        else:
            features_array = np.array(features)

        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        #Predecir
        pred = self.modelo.predict(features_array)[0]
        return NUM_A_JUGADA[pred]

        # features = self.obtener_features_actuales()
        # prediccion = self.modelo.predict([features])[0]
        # return NUM_A_JUGADA[prediccion]



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
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # TODO: Implementa el flujo completo:
    # 1. Cargar datos
    df = cargar_datos()  # usa RUTA_DATOS por defecto
    print(f"[+] Datos cargados: {len(df)} filas")
    # 2. Preparar datos
    df = preparar_datos(df)
    print(f"[+] Datos preparados: {len(df)} filas")
    # 3. Crear features
    df = crear_features(df)
    print(f"[+] Features creadas: {df.shape[1]} columnas")
    # 4. Seleccionar features
    X, y = seleccionar_features(df)
    print(f"[+] Features seleccionadas: {X.shape[1]} columnas")
    # 5. Entrenar modelo
    modelo = entrenar_modelo(X, y)
    print(f"[+] Modelo entrenado: {modelo.__class__.__name__}")
    # 6. Guardar modelo
    joblib.dump(modelo, RUTA_MODELO)

    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()
