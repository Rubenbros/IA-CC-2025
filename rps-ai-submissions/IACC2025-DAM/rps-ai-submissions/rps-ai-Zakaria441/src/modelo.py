"""
RPSAI - Modelo de IA MEJORADO para Piedra, Papel o Tijera
==========================================================
Versión optimizada con mejor detección de patrones
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga los datos del CSV de partidas."""
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta_csv}\n"
            f"Ejecuta primero: python recoger_datos.py"
        )

    df = pd.read_csv(ruta_csv)
    columnas_requeridas = ['numero_ronda', 'jugada_j1', 'jugada_j2']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(f"Faltan columnas: {columnas_faltantes}")

    print(f"✅ Datos cargados: {len(df)} rondas")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos para el modelo."""
    df = df.copy()

    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    def calcular_resultado(row):
        j1 = row['jugada_j1']
        j2 = row['jugada_j2']
        if j1 == j2:
            return 0
        elif GANA_A[j1] == j2:
            return 1
        else:
            return -1

    if 'resultado' not in df.columns:
        df['resultado_num'] = df.apply(calcular_resultado, axis=1)
    else:
        resultado_map = {'victoria': 1, 'empate': 0, 'derrota': -1}
        df['resultado_num'] = df['resultado'].map(resultado_map)

    df = df.dropna(subset=['proxima_jugada_j2'])
    print(f"✅ Datos preparados: {len(df)} rondas válidas")
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURES MEJORADAS - Mejor detección de patrones
    """
    df = df.copy()

    # ========== LAG FEATURES (Memoria más larga) ==========
    for i in range(1, 6):  # Últimas 5 jugadas (antes eran 3)
        df[f'j2_lag{i}'] = df['jugada_j2_num'].shift(i)
        if i <= 3:
            df[f'j1_lag{i}'] = df['jugada_j1_num'].shift(i)

    # ========== RESULTADOS RECIENTES ==========
    for i in range(1, 4):
        df[f'resultado_lag{i}'] = df['resultado_num'].shift(i)

    # ========== FRECUENCIAS EN VENTANAS ==========
    # Ventana corta (últimas 5 rondas)
    for ventana, nombre in [(5, 'corta'), (15, 'media'), (30, 'larga')]:
        for valor, jugada in [(0, 'piedra'), (1, 'papel'), (2, 'tijera')]:
            df[f'freq_j2_{jugada}_{nombre}'] = df['jugada_j2_num'].rolling(
                window=ventana, min_periods=1
            ).apply(lambda x: (x == valor).sum() / len(x))

    # ========== DETECCIÓN DE PATRONES AVANZADA ==========

    # 1. Rachas (cuántas veces seguidas juega lo mismo)
    def calcular_racha(serie):
        rachas = []
        racha_actual = 1
        for i in range(len(serie)):
            if i > 0 and serie.iloc[i] == serie.iloc[i - 1]:
                racha_actual += 1
            else:
                racha_actual = 1
            rachas.append(racha_actual)
        return rachas

    df['racha_j2'] = calcular_racha(df['jugada_j2_num'])

    # 2. ¿Repite la jugada anterior?
    df['j2_repite'] = (df['jugada_j2_num'] == df['j2_lag1']).astype(int)

    # 3. ¿Alterna entre dos opciones? (patrón A-B-A-B)
    df['j2_alterna'] = ((df['jugada_j2_num'] == df['j2_lag2']) &
                        (df['jugada_j2_num'] != df['j2_lag1'])).astype(int)

    # 4. ¿Ciclo de 3? (patrón A-B-C-A-B-C)
    df['j2_ciclo3'] = ((df['jugada_j2_num'] == df['j2_lag3'])).astype(int)

    # 5. ¿Cambia después de perder?
    df['j2_cambia_tras_perder'] = (
            (df['resultado_lag1'] == -1) &
            (df['jugada_j2_num'] != df['j2_lag1'])
    ).astype(int)

    # 6. ¿Repite después de ganar?
    df['j2_repite_tras_ganar'] = (
            (df['resultado_lag1'] == 1) &
            (df['jugada_j2_num'] == df['j2_lag1'])
    ).astype(int)

    # ========== TENDENCIAS ==========
    # Cambios en las frecuencias (está cambiando su estrategia?)
    df['tendencia_piedra'] = (
            df['freq_j2_piedra_corta'] - df['freq_j2_piedra_media']
    )
    df['tendencia_papel'] = (
            df['freq_j2_papel_corta'] - df['freq_j2_papel_media']
    )
    df['tendencia_tijera'] = (
            df['freq_j2_tijera_corta'] - df['freq_j2_tijera_media']
    )

    # ========== CONTEXTO ==========
    df['fase_juego'] = df['numero_ronda'] / df['numero_ronda'].max()

    # Jugada más frecuente reciente
    def jugada_mas_frecuente(serie, ventana=10):
        frecuencias = []
        for i in range(len(serie)):
            inicio = max(0, i - ventana)
            ventana_actual = serie.iloc[inicio:i + 1]
            if len(ventana_actual) > 0:
                moda = ventana_actual.mode()
                frecuencias.append(moda.iloc[0] if len(moda) > 0 else -1)
            else:
                frecuencias.append(-1)
        return frecuencias

    df['j2_favorita_reciente'] = jugada_mas_frecuente(df['jugada_j2_num'], 10)
    df['j2_favorita_global'] = jugada_mas_frecuente(df['jugada_j2_num'], 50)

    # Rellenar NaN
    df = df.fillna(-1)

    print(f"✅ Features creados: {len(df.columns) - 5} características")
    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona las features para entrenar."""

    # Todas las features menos las columnas originales
    excluir = ['jugada_j1', 'jugada_j2', 'timestamp', 'sesion', 'resultado',
               'jugada_j1_num', 'jugada_j2_num', 'proxima_jugada_j2',
               'resultado_num', 'numero_ronda', 'tiempo_reaccion_ms']

    feature_cols = [col for col in df.columns if col not in excluir]

    X = df[feature_cols].values
    y = df['proxima_jugada_j2'].values.astype(int)

    print(f"✅ Features seleccionados: {len(feature_cols)}")
    print(f"   Forma de X: {X.shape}")
    print(f"   Forma de y: {y.shape}")

    return X, y, feature_cols


def entrenar_modelo(X, y, test_size: float = 0.2):
    """Entrena múltiples modelos y selecciona el mejor."""

    print("\n" + "=" * 60)
    print("   ENTRENAMIENTO DE MODELOS MEJORADOS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    print(f"\n📊 Split de datos:")
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test:  {len(X_test)} muestras")

    # Modelos mejorados con mejores hiperparámetros
    modelos = {
        'RandomForest-Profundo': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        'RandomForest-Balanceado': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=3,
            random_state=42
        )
    }

    print(f"\n🧪 Probando {len(modelos)} modelos...\n")

    mejor_score = 0
    mejor_modelo = None
    mejor_nombre = ""

    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")

        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f"  Train Accuracy: {acc_train:.2%}")
        print(f"  Test Accuracy:  {acc_test:.2%}")

        # Penalizar overfitting
        diferencia = acc_train - acc_test
        if diferencia > 0.15:
            print(f"  ⚠️  Posible overfitting (diferencia: {diferencia:.2%})")

        if acc_test > mejor_score:
            mejor_score = acc_test
            mejor_modelo = modelo
            mejor_nombre = nombre

        print()

    print("=" * 60)
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Test Accuracy: {mejor_score:.2%}")
    print("=" * 60)

    # Reporte detallado
    print(f"\n📊 Reporte del mejor modelo:\n")
    y_pred_final = mejor_modelo.predict(X_test)
    print(classification_report(
        y_test, y_pred_final,
        target_names=['Piedra', 'Papel', 'Tijera'],
        zero_division=0
    ))

    # Matriz de confusión
    print("📊 Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred_final)
    print("              Pred:Piedra  Pred:Papel  Pred:Tijera")
    for i, jugada in enumerate(['Real:Piedra', 'Real:Papel', 'Real:Tijera']):
        print(f"{jugada:15s} {cm[i][0]:8d}    {cm[i][1]:8d}    {cm[i][2]:8d}")

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n💾 Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


class JugadorIA:
    """Clase que encapsula el modelo para jugar."""

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠️ Modelo no encontrado. La IA jugará aleatoriamente.")

    def registrar_ronda(self, jugada_humano: str, jugada_ia: str):
        """Registra una ronda jugada."""
        self.historial.append((jugada_humano, jugada_ia))

    def obtener_features_actuales(self) -> np.ndarray:
        """Genera las features basadas en el historial actual."""

        # IMPORTANTE: Debe coincidir con el número de features del entrenamiento
        num_features = 32  # Ajustar según el modelo entrenado

        if len(self.historial) == 0:
            return np.array([-1] * num_features)

        jugadas_humano = [JUGADA_A_NUM[j[0]] for j in self.historial]
        jugadas_ia = [JUGADA_A_NUM[j[1]] for j in self.historial]

        # Calcular resultados
        resultados = []
        for j_humano, j_ia in self.historial:
            if j_ia == j_humano:
                resultados.append(0)
            elif GANA_A[j_ia] == j_humano:
                resultados.append(1)
            else:
                resultados.append(-1)

        features = []

        # Lag features (5 jugadas humano, 3 jugadas IA)
        for i in range(1, 6):
            features.append(jugadas_humano[-i] if len(jugadas_humano) >= i else -1)
        for i in range(1, 4):
            features.append(jugadas_ia[-i] if len(jugadas_ia) >= i else -1)

        # Resultados recientes (3 últimos)
        for i in range(1, 4):
            features.append(resultados[-i] if len(resultados) >= i else 0)

        # Frecuencias en 3 ventanas
        for ventana in [5, 15, 30]:
            ultimas = jugadas_humano[-ventana:]
            total = len(ultimas)
            for valor in [0, 1, 2]:  # piedra, papel, tijera
                freq = ultimas.count(valor) / total if total > 0 else 0
                features.append(freq)

        # Racha actual
        racha = 1
        if len(jugadas_humano) > 1:
            for i in range(len(jugadas_humano) - 2, -1, -1):
                if jugadas_humano[i] == jugadas_humano[-1]:
                    racha += 1
                else:
                    break
        features.append(racha)

        # Patrones
        features.append(1 if len(jugadas_humano) >= 2 and jugadas_humano[-1] == jugadas_humano[-2] else 0)  # repite
        features.append(
            1 if len(jugadas_humano) >= 3 and jugadas_humano[-1] == jugadas_humano[-3] and jugadas_humano[-1] !=
                 jugadas_humano[-2] else 0)  # alterna
        features.append(1 if len(jugadas_humano) >= 4 and jugadas_humano[-1] == jugadas_humano[-4] else 0)  # ciclo3

        # Comportamiento post-resultado
        features.append(1 if len(resultados) >= 2 and resultados[-1] == -1 and jugadas_humano[-1] != jugadas_humano[
            -2] else 0)  # cambia tras perder
        features.append(1 if len(resultados) >= 2 and resultados[-1] == 1 and jugadas_humano[-1] == jugadas_humano[
            -2] else 0)  # repite tras ganar

        # Tendencias (diferencias entre ventanas)
        ultimas_5 = jugadas_humano[-5:]
        ultimas_15 = jugadas_humano[-15:]
        for valor in [0, 1, 2]:
            freq_corta = ultimas_5.count(valor) / len(ultimas_5) if len(ultimas_5) > 0 else 0
            freq_media = ultimas_15.count(valor) / len(ultimas_15) if len(ultimas_15) > 0 else 0
            features.append(freq_corta - freq_media)

        # Fase del juego
        features.append(min(len(self.historial) / 50, 1.0))

        # Jugada favorita
        from collections import Counter
        ultimas_10 = jugadas_humano[-10:]
        if len(ultimas_10) > 0:
            features.append(Counter(ultimas_10).most_common(1)[0][0])
        else:
            features.append(-1)

        ultimas_50 = jugadas_humano[-50:]
        if len(ultimas_50) > 0:
            features.append(Counter(ultimas_50).most_common(1)[0][0])
        else:
            features.append(-1)

        return np.array(features)

    def predecir_jugada_oponente(self) -> str:
        """Predice la próxima jugada del oponente."""
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()
        prediccion = self.modelo.predict([features])[0]
        return NUM_A_JUGADA[int(prediccion)]

    def decidir_jugada(self) -> str:
        """Decide qué jugada hacer para ganar."""
        prediccion_oponente = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[prediccion_oponente]


def main():
    """Función principal para entrenar el modelo."""
    print("=" * 60)
    print("   RPSAI - Entrenamiento MEJORADO")
    print("=" * 60)

    try:
        print("\n[1/5] Cargando datos...")
        df = cargar_datos()

        print("\n[2/5] Preparando datos...")
        df = preparar_datos(df)

        print("\n[3/5] Creando features mejoradas...")
        df = crear_features(df)

        print("\n[4/5] Seleccionando features...")
        X, y, feature_names = seleccionar_features(df)

        print("\n[5/5] Entrenando modelo...")
        modelo = entrenar_modelo(X, y)

        guardar_modelo(modelo)

        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print("\nPróximos pasos:")
        print("  1. Ejecuta: python src/evaluador.py")
        print("  2. Juega contra tu IA mejorada")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()