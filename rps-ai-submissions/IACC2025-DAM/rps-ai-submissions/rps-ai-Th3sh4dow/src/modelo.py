import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")


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
# PARTE 1: EXTRACCION DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    try:
        df = pd.read_csv(ruta_csv)
        required_cols = ['numero_ronda', 'jugada_j1', 'jugada_j2']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: El archivo CSV debe contener las columnas: {required_cols}")
            return pd.DataFrame()
        print(f"Datos cargados correctamente desde: {ruta_csv} ({len(df)} filas)")
        return df
    except FileNotFoundError:
        print(f"Error: Archivo de datos no encontrado en {ruta_csv}. Asegúrate de que existe.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar el CSV: {e}")
        return pd.DataFrame()


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.
    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    """
    if df.empty:
        return df

    # Convierte las jugadas de texto a numeros para J1 y J2
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # Crea la columna 'proxima_jugada_j2' (el target a predecir)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # La última fila tendrá NaN en 'proxima_jugada_j2', la eliminamos
    df.dropna(subset=['proxima_jugada_j2'], inplace=True)

    # Convertir el target a entero (ya que son categorías)
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    print("Datos preparados: Jugadas mapeadas y columna target 'proxima_jugada_j2' creada.")
    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.
    (El cálculo del resultado se mueve al inicio para evitar KeyErrors.)
    """
    df = df.copy()
    if df.empty:
        return df

    # ------------------------------------------
    # Feature 3 - Resultado anterior (CORREGIDO: Calculado primero)
    # ------------------------------------------
    # Codificación: 1=Gana J2, 0=Empate, -1=Pierde J2
    def calcular_resultado(j1_num, j2_num):
        # Manejamos explícitamente NaNs que podrían existir antes de llamar a NUM_A_JUGADA
        if pd.isna(j1_num) or pd.isna(j2_num):
            return np.nan

        j1 = NUM_A_JUGADA[j1_num]
        j2 = NUM_A_JUGADA[j2_num]

        if j1 == j2: return 0  # Empate
        if PIERDE_CONTRA[j1] == j2: return 1  # Gana J2
        return -1  # Pierde J2

    # El 'apply' se ejecuta sobre columnas limpias (jugada_j1_num, jugada_j2_num)
    df['resultado_ronda_num'] = df.apply(
        lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']),
        axis=1
    )

    # Feature final: Resultado de la ronda anterior (shift(1))
    df['resultado_anterior'] = df['resultado_ronda_num'].shift(1)

    # --- Feature Base para el lag: jugada actual del oponente (j2) ---
    j2_num = df['jugada_j2_num']

    # ------------------------------------------
    # Feature 1 - Lag features (jugadas anteriores de j2)
    # ------------------------------------------
    df['j2_lag_1'] = j2_num.shift(1)
    df['j2_lag_2'] = j2_num.shift(2)
    df['j2_lag_3'] = j2_num.shift(3)

    # ------------------------------------------
    # Feature 2 - Frecuencia de la última jugada (se mantiene la lógica original)
    # ------------------------------------------
    for lag in range(1, 4):
        df[f'j2_prob_despues_de_lag_{lag}'] = df.groupby(f'j2_lag_{lag}')['jugada_j2_num'].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    print("Feature Engineering completado: Features de Lag, Frecuencia y Resultado Anterior creadas.")
    return df

def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    Se usan las features más estables: Lag y Resultado Anterior.
    """
    # Usaremos solo Lag y Resultado Anterior para evitar inestabilidades en la predicción en vivo.
    feature_cols = [
        'j2_lag_1', 'j2_lag_2', 'j2_lag_3',  # Lag features
        'resultado_anterior'  # Resultado anterior
    ]

    # Eliminar las filas que tienen NaN debido a los shifts (Lag features y resultado anterior)
    # y asegurarse de que el target no sea NaN.
    df.dropna(subset=feature_cols + ['proxima_jugada_j2'], inplace=True)

    # Crea X (features) e y (target)
    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    print(f"Features seleccionadas: {feature_cols}")
    print(f"Tamaño de los datos para entrenamiento: {len(X)} filas.")

    return X.values, y.values  # Retornamos como numpy arrays (requerido por sklearn)


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.
    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    """
    if X.shape[0] < 5:
        print("Error: No hay suficientes datos para entrenar el modelo.")
        return None

    # Divide los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\nDatos divididos: Train={len(X_train)} Test={len(X_test)}")

    # Entrena varios modelos
    modelos = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    mejor_modelo = None
    mejor_accuracy = -1

    print("\nIniciando entrenamiento y evaluación de modelos...")

    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)

        # Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        print("-" * 40)
        print(f"Modelo: {nombre}")
        print(f"Accuracy: {accuracy:.4f}")

        # Mostrar reporte completo de clasificacion
        try:
            target_names = [NUM_A_JUGADA[i] for i in sorted(np.unique(y))]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        except ValueError:
            print(classification_report(y_test, y_pred, zero_division=0))

        # Seleccionar el mejor
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = modelo

    print("=" * 40)
    print(f"Mejor modelo seleccionado: {type(mejor_modelo).__name__} con Accuracy: {mejor_accuracy:.4f}")
    print("=" * 40)

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
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # Carga el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Modelo de IA cargado correctamente.")
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero (ejecuta el script principal).")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.
        Las features deben coincidir con las usadas en seleccionar_features (Lag 1-3, Resultado Anterior).
        """
        # Si no hay historial, retornamos un array de NaNs.
        if not self.historial:
            # 4 es el número de features: (j2_lag_1, j2_lag_2, j2_lag_3, resultado_anterior)
            return np.array([[np.nan] * 4])

        # 1. Crear DataFrame del historial
        df = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2'])

        # 2. Preparar datos (mapear a números)
        df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
        df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

        # 3. Calcular Features (solo Lag y Resultado Anterior)
        j2_num = df['jugada_j2_num']

        # Lag features
        df['j2_lag_1'] = j2_num.shift(1)
        df['j2_lag_2'] = j2_num.shift(2)
        df['j2_lag_3'] = j2_num.shift(3)

        # Resultado feature
        def calcular_resultado(j1_num, j2_num):
            j1 = NUM_A_JUGADA[j1_num]
            j2 = NUM_A_JUGADA[j2_num]
            if j1 == j2: return 0
            if PIERDE_CONTRA[j1] == j2: return 1
            return -1

        # Calculamos resultado_ronda_num y luego resultado_anterior (el lag 1)
        df['resultado_ronda_num'] = df.apply(
            lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']), axis=1
        )
        df['resultado_anterior'] = df['resultado_ronda_num'].shift(1)

        # Seleccionar solo la última fila (el estado actual) y las features correctas
        feature_cols = ['j2_lag_1', 'j2_lag_2', 'j2_lag_3', 'resultado_anterior']

        # El .iloc[-1] toma la última ronda COMPLETADA, que es la que se usa para predecir la SIGUIENTE.
        features_serie = df.iloc[-1][feature_cols]

        # Retorna el array de forma (1, n_features)
        return features_serie.values.reshape(1, -1)

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        # Si no hay modelo, juega aleatorio (No usamos len < 3 aquí para dar oportunidad al modelo
        # de predecir si es necesario, confiando en la imputación)
        if self.modelo is None or not self.historial:
            return np.random.choice(["piedra", "papel", "tijera"])

        try:
            features = self.obtener_features_actuales()

            # --- CORRECCIÓN CLAVE: Imputación para evitar NaN ---
            # Reemplazar NaNs con 0s.
            # Esto es necesario para las primeras rondas donde los lags son NaN.
            features = np.nan_to_num(features, nan=0.0)

            # Verificamos si, incluso con imputación, solo hay ceros (es decir, muy poco historial)
            if np.all(features == 0):
                # Si el historial es casi nulo, volvemos a jugar aleatorio.
                return np.random.choice(["piedra", "papel", "tijera"])

            prediccion_num = self.modelo.predict(features)[0]
            return NUM_A_JUGADA[prediccion_num]

        except Exception as e:
            # Capturamos el error original (Input X contains NaN) si la imputación falla
            print(f"Advertencia: Fallo en la predicción ({e}). Jugando aleatorio.")
            return np.random.choice(["piedra", "papel", "tijera"])

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        # Juega lo que le gana a la prediccion del oponente
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.
    """
    print("=" * 50)
    print("      RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    # 1. Cargar datos
    print("\n[PASO 1] Cargando datos...")
    df = cargar_datos()
    if df.empty:
        return

    # 2. Preparar datos
    print("\n[PASO 2] Preparando datos...")
    df_preparado = preparar_datos(df)

    # 3. Crear features
    print("\n[PASO 3] Creando features...")
    df_features = crear_features(df_preparado)

    # 4. Seleccionar features
    print("\n[PASO 4] Seleccionando features...")
    X, y = seleccionar_features(df_features)

    if len(X) == 0:
        print("Error: No hay suficientes datos limpios para entrenar después de crear features.")
        return

    # 5. Entrenar modelo
    print("\n[PASO 5] Entrenando modelo(s)...")
    modelo_entrenado = entrenar_modelo(X, y)

    if modelo_entrenado:
        # 6. Guardar modelo
        print("\n[PASO 6] Guardando el mejor modelo...")
        guardar_modelo(modelo_entrenado)
    else:
        print("El entrenamiento del modelo falló o no se pudo seleccionar un modelo.")


if __name__ == "__main__":
    main()