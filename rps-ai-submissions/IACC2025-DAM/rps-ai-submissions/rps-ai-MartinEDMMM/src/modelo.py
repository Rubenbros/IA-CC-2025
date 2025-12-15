import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np


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


def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    try:
        df = pd.read_csv(ruta_csv)

        # Verificar columnas mínimas requeridas
        if not all(col in df.columns for col in ['numero_ronda', 'jugada_j1', 'jugada_j2']):
            raise ValueError("El CSV no contiene las columnas requeridas (numero_ronda, jugada_j1, jugada_j2).")

        print(f"Datos cargados correctamente desde: {ruta_csv}")
        return df

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo CSV en: {ruta_csv}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR al cargar datos: {e}")
        return pd.DataFrame()


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.
    """
    if df.empty:
        return df

    df = df.copy()


    # 1. Convierte las jugadas de texto a numeros
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # 2. Crea la columna 'proxima_jugada_j2' (el target a predecir)
    # Target: la jugada del oponente (j2) en la siguiente ronda.
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # 3. Elimina la última fila que tendrá NaN en 'proxima_jugada_j2'
    df.dropna(subset=['proxima_jugada_j2'], inplace=True)
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    print("Preparación de datos completada: Jugadas a números y Target creado.")
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.
    Se implementan 3 tipos de features: Lag, Frecuencia y Resultado Anterior.
    """
    df = df.copy()
    if df.empty:
        return df


    j2_num = df['jugada_j2_num']

    # ------------------------------------------
    # ------------------------------------------
    # Las últimas 3 jugadas del oponente (shift(1), shift(2), shift(3))
    df['j2_lag_1'] = j2_num.shift(1)
    df['j2_lag_2'] = j2_num.shift(2)
    df['j2_lag_3'] = j2_num.shift(3)


    for lag in range(1, 4):
        # La jugada actual es el resultado de la transición de la jugada 'lag'
        df[f'j2_prob_despues_de_lag_{lag}'] = df.groupby(f'j2_lag_{lag}')['jugada_j2_num'].transform(
            lambda x: x.expanding().mean().shift(1))


    # Codificación: 1=Gana J2, 0=Empate, -1=Pierde J2
    def calcular_resultado(j1, j2):
        if j1 == j2: return 0  # Empate
        if PIERDE_CONTRA[NUM_A_JUGADA[j1]] == NUM_A_JUGADA[j2]: return 1  # Gana J2 (ej: j1=piedra, j2=papel -> j2 gana)
        return -1  # Pierde J2

    df['resultado_ronda_num'] = df.apply(lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']),
                                         axis=1)

    # Feature final: Resultado de la ronda anterior (shift(1))
    df['resultado_anterior'] = df['resultado_ronda_num'].shift(1)

    print("Feature Engineering completado: Features de Lag, Frecuencia y Resultado Anterior creadas.")
    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    """
    # 1. Eliminar las filas que tienen NaN debido a los shifts (Lag features y resultado anterior)
    df.dropna(inplace=True)

    # Usaremos las features de Lag y la de Resultado Anterior (las de frecuencia no son estables con 'expanding')
    feature_cols = [
        'j2_lag_1', 'j2_lag_2', 'j2_lag_3',  # Lag features
        'resultado_anterior'  # Resultado anterior
    ]



    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    print(f"Features seleccionadas: {feature_cols}")
    print(f"Tamaño de los datos para entrenamiento: {len(X)} filas.")

    return X.values, y.values  # Retornamos como numpy arrays (requerido por sklearn)


def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.
    """
    mejor_modelo = None
    mejor_accuracy = 0


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )


    modelos = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("\n--- Evaluación de Modelos ---")

    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)

        # Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[Modelo: {nombre}]")
        print(f"Accuracy (Precisión): {accuracy:.4f}")
        print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

        # Seleccionar el mejor
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = modelo

    print(f"\n--- Selección Final ---")
    print(f"Mejor modelo: {type(mejor_modelo).__name__} con Accuracy: {mejor_accuracy:.4f}")


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



class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        # Almacenamos solo las jugadas numéricas para facilitar el cálculo de features
        self.historial_num = []  # Lista de (jugada_j1_num, jugada_j2_num)


        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Jugador IA inicializado con modelo entrenado.")
        except FileNotFoundError:
            print("ADVERTENCIA: Modelo no encontrado. La IA jugará aleatorio hasta ser entrenada.")

    def _jugar_aleatorio(self):
        """Helper para jugar aleatorio si no hay modelo."""
        return np.random.choice(["piedra", "papel", "tijera"])

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.
        """
        jugada_j1_num = JUGADA_A_NUM.get(jugada_j1)
        jugada_j2_num = JUGADA_A_NUM.get(jugada_j2)

        if jugada_j1_num is not None and jugada_j2_num is not None:
            self.historial_num.append((jugada_j1_num, jugada_j2_num))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.
        """
        if len(self.historial_num) < 4:  # Necesita al menos 3 rondas para el lag_3 y el resultado anterior
            return None


        # Extraer las jugadas de j2 (oponente) de la última parte del historial
        jugadas_j2 = [j2 for j1, j2 in self.historial_num]

        # 1. Lag features (solo necesitamos el último valor de los shifts)
        # La jugada actual del oponente fue jugadas_j2[-1]
        j2_lag_1 = jugadas_j2[-1]
        j2_lag_2 = jugadas_j2[-2]
        j2_lag_3 = jugadas_j2[-3]

        # 2. Resultado anterior (comparar la penúltima jugada)
        ultima_ronda_j1 = self.historial_num[-1][0]
        ultima_ronda_j2 = self.historial_num[-1][1]

        def calcular_resultado(j1, j2):
            # 1=Gana J2, 0=Empate, -1=Pierde J2
            if j1 == j2: return 0
            # Convertimos a texto para usar la regla GANA_A / PIERDE_CONTRA
            if PIERDE_CONTRA[NUM_A_JUGADA[j1]] == NUM_A_JUGADA[j2]: return 1
            return -1

        # El resultado anterior es el resultado de la ÚLTIMA ronda registrada
        resultado_anterior = calcular_resultado(ultima_ronda_j1, ultima_ronda_j2)


        features = np.array([
            j2_lag_1,
            j2_lag_2,
            j2_lag_3,
            resultado_anterior
        ]).reshape(1, -1)

        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        if self.modelo is None or len(self.historial_num) < 4:
            # Si no hay modelo entrenado o no hay suficiente historial, juega aleatorio
            return self._jugar_aleatorio()


        features = self.obtener_features_actuales()

        if features is None:
            return self._jugar_aleatorio()

        # El modelo espera un array 2D
        prediccion_num = self.modelo.predict(features)[0]

        # Convierte la predicción numérica a texto
        return NUM_A_JUGADA[prediccion_num]

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]



def main():
    """
    Funcion principal para entrenar el modelo.
    """
    print("=" * 50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    # 1. Cargar datos
    df_raw = cargar_datos()
    if df_raw.empty:
        return

    # 2. Preparar datos
    df_prepared = preparar_datos(df_raw)

    # 3. Crear features
    df_features = crear_features(df_prepared)

    # 4. Seleccionar features
    X, y = seleccionar_features(df_features)

    # 5. Entrenar modelo
    modelo_entrenado = entrenar_modelo(X, y)

    # 6. Guardar modelo
    if modelo_entrenado:
        guardar_modelo(modelo_entrenado)


if __name__ == "__main__":
    main()