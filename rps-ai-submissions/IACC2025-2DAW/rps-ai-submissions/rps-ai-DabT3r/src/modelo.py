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


import pandas as pd
import os
import pickle
import warnings
from pathlib import Path


import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "Datos.csv"
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

    TODO: Implementa esta funcion
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

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"{ruta_csv} no existe")

    df = pd.read_csv(ruta_csv)

    # TODO: Implementa la carga de datos
    # Pista: usa pd.read_csv()
    df = df.rename(columns={
        'ronda': 'numero_ronda',
        'player1_move': 'jugada_j1',
        'player2_move': 'jugada_j2'
    })

    columnas_requeridas = ['numero_ronda', 'jugada_j1', 'jugada_j2']
    for a in columnas_requeridas:
        if a not in df.columns:
            raise ValueError(f"{a} no existe")

    print(f"CSV cargado correctamente. Columnas: {list(df.columns)}")
    print(f"Primeras filas:")
    print(df.head())
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    TODO: Implementa esta funcion
    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    print("Convirtiendo jugadas de texto a numeros...")
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    print("Creando columna target 'proxima_jugada_j2'...")
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    print("Eliminando filas con valores nulos...")
    filas_antes = len(df)
    df = df.dropna()
    filas_despues = len(df)

    print(f"Filas antes de limpiar: {filas_antes}")
    print(f"Filas despues de limpiar: {filas_despues}")
    print(f"Filas eliminadas: {filas_antes - filas_despues}")

    print("\nDatos preparados (primeras 5 filas):")
    columnas_mostrar = ['numero_ronda', 'jugada_j1', 'jugada_j2', 'jugada_j1_num', 'jugada_j2_num', 'proxima_jugada_j2']
    print(df[columnas_mostrar].head())

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    TODO: Implementa al menos 3 tipos de features diferentes.

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
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    print("Creando Feature 1: Frecuencia de jugadas...")
    for a in ['piedra', 'papel', 'tijera']:
        df[f'frecuencia_{a}_j2'] = (df['jugada_j2'] == a).expanding().mean()
    print("Frecuencias creadas: frecuencia_piedra_j2, frecuencia_papel_j2, frecuencia_tijera_j2")

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    print("Creando Feature 2: Lag features...")
    df['jugada_j2_anterior1'] = df['jugada_j2_num'].shift(1)
    df['jugada_j2_anterior2'] = df['jugada_j2_num'].shift(2)
    df['jugada_j2_anterior3'] = df['jugada_j2_num'].shift(3)
    df['jugada_j1_anterior1'] = df['jugada_j1_num'].shift(1)
    print("Lag features creadas: jugada_j2_anterior1, jugada_j2_anterior2, jugada_j2_anterior3, jugada_j1_anterior1")

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    print("Creando Feature 3: Resultado anterior...")

    def calcular_resultado(j1_numero, j2_numero):
        if j1_numero == j2_numero:
            return 0
        elif (j1_numero == 0 and j2_numero == 2) or (j1_numero == 1 and j2_numero == 0) or (
                j1_numero == 2 and j2_numero == 1):
            return 1
        else:
            return -1

    df['resultado_ronda'] = df.apply(
        lambda celda: calcular_resultado(celda['jugada_j1_num'], celda['jugada_j2_num']), axis=1
    )
    df['resultado_anterior'] = df['resultado_ronda'].shift(1)
    print("Resultado anterior creado: resultado_anterior")

    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    print("Creando Feature 4: Racha actual...")
    df['victoria_j2'] = (df['resultado_ronda'] == -1).astype(int)
    df['racha_victoria_j2'] = df['victoria_j2'].groupby(
        (df['victoria_j2'] == 0).cumsum()
    ).cumsum()
    print("Racha creada: racha_victoria_j2")

    print("Creando Feature 5: Patrones despues de ganar/perder...")
    df['cambio_jugada_j2'] = (df['jugada_j2_num'] != df['jugada_j2_num'].shift(1)).astype(int)
    df['patron_ganar_cambiar'] = ((df['resultado_anterior'] == -1) & (df['cambio_jugada_j2'] == 1)).astype(int)
    df['patron_perder_cambiar'] = ((df['resultado_anterior'] == 1) & (df['cambio_jugada_j2'] == 1)).astype(int)
    print("Patrones creados: cambio_jugada_j2, patron_ganar_cambiar, patron_perder_cambiar")

    filas_antes = len(df)
    df = df.dropna()
    filas_despues = len(df)

    print("\nResumen de features creadas:")
    print(
        f"   Features totales: {len([col for col in df.columns if col not in ['numero_ronda', 'jugada_j1', 'jugada_j2', 'jugada_j1_num', 'jugada_j2_num', 'proxima_jugada_j2']])}")
    print(f"   Filas antes/despues: {filas_antes}/{filas_despues}")
    print(f"   Columnas disponibles: {list(df.columns)}")

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    TODO: Implementa esta funcion
    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """
    # TODO: Selecciona las columnas de features
    print(f"Seleccionando features...")
    feature_cols = [
        'frecuencia_piedra_j2', 'frecuencia_papel_j2', 'frecuencia_tijera_j2',
        'jugada_j2_anterior1', 'jugada_j2_anterior2', 'jugada_j2_anterior3', 'jugada_j1_anterior1',
        'resultado_anterior',
        'racha_victoria_j2', 'cambio_jugada_j2',
        'patron_ganar_cambiar', 'patron_perder_cambiar'
    ]

    print(f"   {len(feature_cols)} features seleccionadas:")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {col}")

    # TODO: Crea X (features) e y (target)
    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    print(f"\nDimensiones finales:")
    print(f"   X (features): {X.shape}")
    print(f"   y (target): {y.shape}")
    print(f"   Target values: {y.unique()} -> {[NUM_A_JUGADA[v] for v in y.unique()]}")

    # Elimina filas con valores nulos
    if X.isnull().any().any() or y.isnull().any():
        print("Eliminando filas con valores nulos...")
        mask = ~X.isnull().any(axis=1) & ~y.isnull()
        X = X[mask]
        y = y[mask]
        print(f"   Filas despues de limpiar: {len(X)}")

    return X, y


def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    TODO: Implementa esta funcion
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
    # TODO: Divide los datos
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"   Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   Prueba: {X_test.shape[0]} muestras")

    # TODO: Entrena varios modelos
    print("\nInicializando modelos...")
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    mejor_modelo = None
    mejor_accuracy = 0
    mejor_nombre = ""

    # TODO: Evalua cada modelo
    print("\nEntrenando y evaluando modelos...")
    for nombre, modelo in modelos.items():
        print(f"\n--- {nombre} ---")

        # Entrena con fit()
        modelo.fit(X_train, y_train)

        # Predice con predict()
        y_pred = modelo.predict(X_test)

        # Calcula accuracy con accuracy_score()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Accuracy: {accuracy:.3f}")

        # Muestra classification_report()
        print("   Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['piedra', 'papel', 'tijera']))

        # Matriz de confusion adicional
        cm = confusion_matrix(y_test, y_pred)
        print(f"   Matriz de confusion:")
        print(f"   {cm}")

        # TODO: Selecciona y retorna el mejor modelo
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = modelo
            mejor_nombre = nombre
            print(f"   Nuevo mejor modelo!")

    print(f"\nMEJOR MODELO SELECCIONADO: {mejor_nombre}")
    print(f"   Accuracy: {mejor_accuracy:.3f}")

    # Re-entrenar con todos los datos
    print(f"\nRe-entrenando mejor modelo con todos los datos...")
    mejor_modelo.fit(X, y)

    return mejor_modelo


# =============================================================================
# PARTE 2: FUNCIONES PARA MODELOS
# =============================================================================

def cargar_modelo(ruta_modelo=None):
    """
    Carga un modelo entrenado desde un archivo.

    Args:
        ruta_modelo: Ruta al archivo del modelo (usa RUTA_MODELO por defecto)

    Returns:
        Modelo cargado

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si no se puede cargar el modelo
    """
    if ruta_modelo is None:
        ruta_modelo = RUTA_MODELO

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"El archivo {ruta_modelo} no existe")

    try:
        import joblib
        with open(ruta_modelo, 'rb') as archivo:
            modelo = joblib.load(archivo)
        return modelo
    except:
        try:
            with open(ruta_modelo, 'rb') as archivo:
                modelo = pickle.load(archivo)
            return modelo
        except Exception as e:
            raise ValueError(f"No se pudo cargar el modelo: {e}")

def guardar_modelo(modelo, ruta_modelo=None):
    """
    Guarda un modelo entrenado en un archivo.

    Args:
        modelo: Modelo entrenado a guardar
        ruta_modelo: Ruta donde guardar el modelo (usa RUTA_MODELO por defecto)

    Raises:
        ValueError: Si no se puede guardar el modelo
    """
    if ruta_modelo is None:
        ruta_modelo = RUTA_MODELO

    directorio = Path(ruta_modelo).parent
    directorio.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        with open(ruta_modelo, 'wb') as archivo:
            joblib.dump(modelo, archivo)
    except:
        try:
            with open(ruta_modelo, 'wb') as archivo:
                pickle.dump(modelo, archivo)
        except Exception as e:
            raise ValueError(f"No se pudo guardar el modelo: {e}")


# =============================================================================
# PARTE 3: CLASE JUGADOR IA (40% de la nota)
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
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # TODO: Carga el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Modelo cargado correctamente")
        except FileNotFoundError:
            print("Modelo no encontrado. Jugara de forma aleatoria.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))
        print(f"Ronda registrada: {jugada_j1} vs {jugada_j2}")
        print(f"   Historial: {len(self.historial)} rondas")

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        TODO: Implementa esta funcion
        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion
        """
        if len(self.historial) < 3:
            print("Historial insuficiente, usando features por defecto")
            return np.array([0.33, 0.33, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        temp_df = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2'])
        temp_df['jugada_j1_num'] = temp_df['jugada_j1'].map(JUGADA_A_NUM)
        temp_df['jugada_j2_num'] = temp_df['jugada_j2'].map(JUGADA_A_NUM)

        features = []
        # TODO: Calcula las features basadas en self.historial
        # Deben ser LAS MISMAS features que usaste para entrenar
        for jugada in ['piedra', 'papel', 'tijera']:
            freq = (temp_df['jugada_j2'] == jugada).mean()
            features.append(freq)

        features.append(temp_df['jugada_j2_num'].iloc[-1] if len(temp_df) > 0 else 0)
        features.append(temp_df['jugada_j2_num'].iloc[-2] if len(temp_df) > 1 else 0)
        features.append(temp_df['jugada_j2_num'].iloc[-3] if len(temp_df) > 2 else 0)
        features.append(temp_df['jugada_j1_num'].iloc[-1] if len(temp_df) > 0 else 0)

        if len(temp_df) > 0:
            ult_j1 = temp_df['jugada_j1_num'].iloc[-1]
            ult_j2 = temp_df['jugada_j2_num'].iloc[-1]
            if ult_j1 == ult_j2:
                features.append(0)
            elif (ult_j1 == 0 and ult_j2 == 2) or (ult_j1 == 1 and ult_j2 == 0) or (ult_j1 == 2 and ult_j2 == 1):
                features.append(1)
            else:
                features.append(-1)
        else:
            features.append(0)

        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)

        #print(f"Features generadas: {features}")
        return np.array(features)

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        TODO: Implementa esta funcion
        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None or len(self.historial) < 2:
            print("Sin modelo o historial insuficiente, jugando aleatorio")
            return np.random.choice(["piedra", "papel", "tijera"])

        # TODO: Implementa la prediccion
        features = self.obtener_features_actuales()
        prediccion = self.modelo.predict([features])[0]
        jugada_predicha = NUM_A_JUGADA[prediccion]

        #print(f"IA predice que el oponente jugara: {jugada_predicha}")
        return jugada_predicha

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            print("Prediccion nula, jugando aleatorio")
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        jugada_ganadora = PIERDE_CONTRA[prediccion_oponente]
        #print(f"IA jugara: {jugada_ganadora} (gana a {prediccion_oponente})")
        return jugada_ganadora

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

    try:
        # 1. Cargar datos
        print("\n1. Cargando datos...")
        df = cargar_datos()
        print(f"   Datos cargados: {len(df)} rondas")

        # 2. Preparar datos
        print("\n2. Preparando datos...")
        df_preparado = preparar_datos(df)
        print(f"   Datos preparados: {len(df_preparado)} rondas validas")

        # 3. Crear features
        print("\n3. Creando features...")
        df_features = crear_features(df_preparado)
        print(f"   Features creadas: {len(df_features.columns)} columnas")

        # 4. Seleccionar features
        print("\n4. Seleccionando features...")
        X, y = seleccionar_features(df_features)
        print(f"   Features seleccionadas: {X.shape[1]} caracteristicas, {X.shape[0]} muestras")

        # 5. Entrenar modelo
        print("\n5. Entrenando modelo...")
        mejor_modelo = entrenar_modelo(X, y)

        # 6. Guardar modelo
        print("\n6. Guardando modelo...")
        guardar_modelo(mejor_modelo)

        print("\n" + "=" * 50)
        print("   ENTRENAMIENTO COMPLETADO")
        print("=" * 50)
        print("\nAhora puedes usar la IA para jugar:")
        print("   ia = JugadorIA()")
        print("   jugada = ia.decidir_jugada()")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nAsegurate de:")
        print("   1. Tener el archivo data/Datos.csv")
        print("   2. El CSV tiene las columnas: numero_ronda,jugada_j1,jugada_j2")
        print("   3. Las jugadas son: piedra, papel o tijera")


if __name__ == "__main__":
    main()