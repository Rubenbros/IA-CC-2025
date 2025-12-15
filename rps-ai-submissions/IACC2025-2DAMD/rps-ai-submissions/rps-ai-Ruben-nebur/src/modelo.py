"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================
Implementación completa del modelo predictivo
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Suprimir warnings de sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

# Importar modelos
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Configuración de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"
RUTA_SCALER = RUTA_PROYECTO / "models" / "scaler.pkl"

# Mapeo de jugadas a números
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Reglas del juego
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCIÓN DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    Args:
        ruta_csv: Ruta al archivo CSV

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    # Verificar que existe el archivo
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    # Cargar datos - tu CSV tiene headers
    df = pd.read_csv(ruta_csv)

    # Renombrar columnas al formato esperado
    df.columns = ['numero_ronda', 'jugada_j1', 'jugada_j2',
                  'resultado', 'tiempo_reaccion', 'timestamp']

    # Convertir a minúsculas las jugadas para consistencia
    df['jugada_j1'] = df['jugada_j1'].str.lower()
    df['jugada_j2'] = df['jugada_j2'].str.lower()
    df['resultado'] = df['resultado'].str.lower()

    print(f"✓ Datos cargados: {len(df)} rondas")
    print(f"  Columnas: {list(df.columns)}")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    # Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # Crear columna de próxima jugada (target)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Crear columna de resultado desde la perspectiva de J2
    def resultado_j2(resultado):
        if 'jugador2' in resultado:
            return 1  # J2 ganó
        elif 'empate' in resultado:
            return 0  # Empate
        else:
            return -1  # J2 perdió

    df['resultado_j2'] = df['resultado'].apply(resultado_j2)

    # Eliminar última fila (no tiene próxima jugada)
    df = df[:-1]

    print(f"✓ Datos preparados: {len(df)} filas válidas")

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features avanzadas para el modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # ------------------------------------------
    # Feature 1: Frecuencia de jugadas de J2
    # ------------------------------------------
    # Frecuencia histórica acumulada de cada jugada
    df['freq_piedra_j2'] = (df['jugada_j2'] == 'piedra').expanding().mean()
    df['freq_papel_j2'] = (df['jugada_j2'] == 'papel').expanding().mean()
    df['freq_tijera_j2'] = (df['jugada_j2'] == 'tijera').expanding().mean()

    # ------------------------------------------
    # Feature 2: Lag features (últimas jugadas)
    # ------------------------------------------
    # Últimas 3 jugadas de J2
    df['lag1_j2'] = df['jugada_j2_num'].shift(1)
    df['lag2_j2'] = df['jugada_j2_num'].shift(2)
    df['lag3_j2'] = df['jugada_j2_num'].shift(3)

    # Últimas 2 jugadas de J1 (puede influir en J2)
    df['lag1_j1'] = df['jugada_j1_num'].shift(1)
    df['lag2_j1'] = df['jugada_j1_num'].shift(2)

    # ------------------------------------------
    # Feature 3: Resultado anterior
    # ------------------------------------------
    df['resultado_anterior'] = df['resultado_j2'].shift(1)

    # ------------------------------------------
    # Feature 4: Racha actual de J2
    # ------------------------------------------
    # Calcular racha de victorias/derrotas
    resultado_anterior = df['resultado_j2'].shift(1).fillna(0).values
    racha_actual = 0
    rachas = []

    for i in range(len(df)):
        val = resultado_anterior[i]
        if val == 1:  # Ganó
            racha_actual = max(1, racha_actual + 1) if racha_actual >= 0 else 1
        elif val == -1:  # Perdió
            racha_actual = min(-1, racha_actual - 1) if racha_actual <= 0 else -1
        else:  # Empate o NaN
            racha_actual = 0
        rachas.append(racha_actual)

    df['racha_j2'] = rachas

    # ------------------------------------------
    # Feature 5: Patrón post-resultado
    # ------------------------------------------
    # Qué juega J2 después de ganar/perder
    df['juega_despues_ganar'] = np.where(
        df['resultado_anterior'] == 1,
        df['jugada_j2_num'],
        np.nan
    )
    df['patron_post_win'] = df['juega_despues_ganar'].expanding().mean()

    df['juega_despues_perder'] = np.where(
        df['resultado_anterior'] == -1,
        df['jugada_j2_num'],
        np.nan
    )
    df['patron_post_loss'] = df['juega_despues_perder'].expanding().mean()

    # ------------------------------------------
    # Feature 6: Tiempo de reacción
    # ------------------------------------------
    df['tiempo_reaccion_norm'] = df['tiempo_reaccion'] / df['tiempo_reaccion'].mean()
    df['tiempo_anterior'] = df['tiempo_reaccion'].shift(1)

    # ------------------------------------------
    # Feature 7: Fase del juego
    # ------------------------------------------
    total_rondas = df['numero_ronda'].max()
    df['fase_juego'] = pd.cut(
        df['numero_ronda'],
        bins=[0, total_rondas*0.33, total_rondas*0.66, total_rondas],
        labels=[0, 1, 2]  # inicio, medio, final
    ).astype(float)

    # ------------------------------------------
    # Feature 8: ¿Repite jugada?
    # ------------------------------------------
    df['j2_repite'] = (df['jugada_j2_num'] == df['lag1_j2']).astype(int)
    df['tasa_repeticion'] = df['j2_repite'].expanding().mean()

    # Rellenar NaN con 0 o valores por defecto
    df = df.fillna(0)

    print(f"✓ Features creadas: {len([c for c in df.columns if c not in ['jugada_j1', 'jugada_j2', 'resultado', 'timestamp']])} features")

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    Returns:
        (X, y) - Features y target
    """
    # Definir columnas de features
    feature_cols = [
        # Frecuencias
        'freq_piedra_j2', 'freq_papel_j2', 'freq_tijera_j2',
        # Lags
        'lag1_j2', 'lag2_j2', 'lag3_j2',
        'lag1_j1', 'lag2_j1',
        # Resultados y rachas
        'resultado_anterior', 'racha_j2',
        # Patrones
        'patron_post_win', 'patron_post_loss',
        # Tiempo
        'tiempo_reaccion_norm', 'tiempo_anterior',
        # Otros
        'fase_juego', 'tasa_repeticion',
        'numero_ronda'
    ]

    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    # Eliminar filas con NaN en target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    print(f"✓ Features seleccionadas: {len(feature_cols)}")
    print(f"  Muestras totales: {len(X)}")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena múltiples modelos y selecciona el mejor.

    Args:
        X: Features
        y: Target
        test_size: Proporción de datos para test

    Returns:
        Mejor modelo entrenado y scaler
    """
    print("\n" + "="*50)
    print("   ENTRENAMIENTO DE MODELOS")
    print("="*50)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    print(f"\n✓ Datos divididos:")
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Test:  {len(X_test)} muestras")

    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir modelos a probar
    modelos = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    resultados = {}

    # Entrenar y evaluar cada modelo
    print("\n" + "-"*50)
    print("EVALUACIÓN DE MODELOS:")
    print("-"*50)

    for nombre, modelo in modelos.items():
        print(f"\n{nombre}:")

        # Entrenar
        modelo.fit(X_train_scaled, y_train)

        # Predecir
        y_pred_train = modelo.predict(X_train_scaled)
        y_pred_test = modelo.predict(X_test_scaled)

        # Métricas
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'modelo': modelo,
            'acc_train': acc_train,
            'acc_test': acc_test
        }

        print(f"  Accuracy Train: {acc_train:.3f}")
        print(f"  Accuracy Test:  {acc_test:.3f}")

        # Mostrar reporte detallado solo para el test
        if acc_test > 0.3:  # Solo si tiene rendimiento decente
            print("\n  Classification Report (Test):")
            report = classification_report(
                y_test, y_pred_test,
                target_names=['Piedra', 'Papel', 'Tijera'],
                zero_division=0
            )
            print("  " + report.replace("\n", "\n  "))

    # Seleccionar mejor modelo (por accuracy en test)
    mejor_nombre = max(resultados.items(), key=lambda x: x[1]['acc_test'])[0]
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_acc = resultados[mejor_nombre]['acc_test']

    print("\n" + "="*50)
    print(f"✓ MEJOR MODELO: {mejor_nombre}")
    print(f"  Accuracy: {mejor_acc:.3f}")
    print("="*50)

    # Matriz de confusión del mejor modelo
    y_pred_final = mejor_modelo.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_final)
    print("\nMatriz de Confusión (Test):")
    print("            Pred:")
    print("          P  Pa  T")
    for i, row in enumerate(cm):
        print(f"Real {['P ', 'Pa', 'T '][i]}: {row}")

    return mejor_modelo, scaler


def guardar_modelo(modelo, scaler, ruta_modelo: str = None, ruta_scaler: str = None):
    """Guarda el modelo y el scaler entrenados."""
    if ruta_modelo is None:
        ruta_modelo = RUTA_MODELO
    if ruta_scaler is None:
        ruta_scaler = RUTA_SCALER

    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)

    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo, f)

    with open(ruta_scaler, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Modelo guardado en: {ruta_modelo}")
    print(f"✓ Scaler guardado en: {ruta_scaler}")


def cargar_modelo(ruta_modelo: str = None, ruta_scaler: str = None):
    """Carga un modelo y scaler previamente entrenados."""
    if ruta_modelo is None:
        ruta_modelo = RUTA_MODELO
    if ruta_scaler is None:
        ruta_scaler = RUTA_SCALER

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")

    with open(ruta_modelo, "rb") as f:
        modelo = pickle.load(f)

    with open(ruta_scaler, "rb") as f:
        scaler = pickle.load(f)

    return modelo, scaler


# =============================================================================
# PARTE 4: PREDICCIÓN Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar contra oponentes.
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.scaler = None
        self.historial = []  # Lista de diccionarios con info de cada ronda

        try:
            self.modelo, self.scaler = cargar_modelo(ruta_modelo)
            print("✓ Modelo cargado exitosamente")
        except FileNotFoundError:
            print("⚠ Modelo no encontrado. Entrena primero con: python src/modelo.py")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str, tiempo: float = 0):
        """
        Registra una ronda jugada.

        Args:
            jugada_j1: Jugada del jugador 1 (IA)
            jugada_j2: Jugada del oponente
            tiempo: Tiempo de reacción
        """
        # Determinar resultado desde perspectiva de J2
        if GANA_A[jugada_j2] == jugada_j1:
            resultado = 1  # J2 ganó
        elif GANA_A[jugada_j1] == jugada_j2:
            resultado = -1  # J2 perdió
        else:
            resultado = 0  # Empate

        self.historial.append({
            'jugada_j1': jugada_j1,
            'jugada_j2': jugada_j2,
            'resultado_j2': resultado,
            'tiempo': tiempo
        })



    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        Returns:
            Array con las features para predicción
        """
        if len(self.historial) == 0:
            # Primera jugada: features por defecto
            return np.array([0.33, 0.33, 0.33,  # frecuencias
                           0.0, 0.0, 0.0, 0.0, 0.0,  # lags
                           0.0, 0.0,  # resultado y racha
                           1.0, 1.0,  # patrones
                           1.0, 3.0,  # tiempos (norm=1, anterior=3)
                           0.0, 0.0,  # fase y tasa
                           1.0])  # ronda

        # Convertir historial a arrays
        jugadas_j1 = [JUGADA_A_NUM[h['jugada_j1']] for h in self.historial]
        jugadas_j2 = [JUGADA_A_NUM[h['jugada_j2']] for h in self.historial]
        resultados = [h['resultado_j2'] for h in self.historial]
        tiempos = [h['tiempo'] if h['tiempo'] > 0 else 3.0 for h in self.historial]

        n_rondas = len(self.historial)

        # Feature 1: Frecuencias
        freq_piedra = sum(1 for j in jugadas_j2 if j == 0) / n_rondas
        freq_papel = sum(1 for j in jugadas_j2 if j == 1) / n_rondas
        freq_tijera = sum(1 for j in jugadas_j2 if j == 2) / n_rondas

        # Feature 2: Lags
        lag1_j2 = float(jugadas_j2[-1]) if n_rondas >= 1 else 0.0
        lag2_j2 = float(jugadas_j2[-2]) if n_rondas >= 2 else 0.0
        lag3_j2 = float(jugadas_j2[-3]) if n_rondas >= 3 else 0.0
        lag1_j1 = float(jugadas_j1[-1]) if n_rondas >= 1 else 0.0
        lag2_j1 = float(jugadas_j1[-2]) if n_rondas >= 2 else 0.0

        # Feature 3: Resultado anterior y racha
        resultado_anterior = float(resultados[-1]) if n_rondas >= 1 else 0.0

        # Calcular racha
        racha = 0.0
        for i in range(len(resultados)-1, -1, -1):
            if resultados[i] == 1 and racha >= 0:
                racha += 1
            elif resultados[i] == -1 and racha <= 0:
                racha -= 1
            else:
                break

        # Feature 4: Patrones post resultado
        jugadas_post_win = [jugadas_j2[i] for i in range(1, n_rondas)
                           if resultados[i-1] == 1]
        patron_post_win = float(np.mean(jugadas_post_win)) if jugadas_post_win else 1.0

        jugadas_post_loss = [jugadas_j2[i] for i in range(1, n_rondas)
                            if resultados[i-1] == -1]
        patron_post_loss = float(np.mean(jugadas_post_loss)) if jugadas_post_loss else 1.0

        # Feature 5: Tiempos (con manejo seguro de división por cero)
        tiempo_medio = float(np.mean(tiempos)) if tiempos else 3.0
        if tiempo_medio == 0 or not np.isfinite(tiempo_medio):
            tiempo_medio = 3.0

        tiempo_actual = tiempos[-1] if tiempos else 3.0
        tiempo_norm = tiempo_actual / tiempo_medio if tiempo_medio > 0 else 1.0
        tiempo_anterior = float(tiempos[-2]) if len(tiempos) >= 2 else 3.0

        # Feature 6: Fase del juego (estimamos 30 rondas como máximo)
        fase = 0.0 if n_rondas < 10 else (1.0 if n_rondas < 20 else 2.0)

        # Feature 7: Tasa de repetición
        repeticiones = sum(1 for i in range(1, n_rondas)
                          if jugadas_j2[i] == jugadas_j2[i-1])
        tasa_repeticion = float(repeticiones) / (n_rondas - 1) if n_rondas > 1 else 0.0

        # Construir array de features (asegurar que todos son float)
        features = np.array([
            freq_piedra, freq_papel, freq_tijera,
            lag1_j2, lag2_j2, lag3_j2,
            lag1_j1, lag2_j1,
            resultado_anterior, racha,
            patron_post_win, patron_post_loss,
            tiempo_norm, tiempo_anterior,
            fase, tasa_repeticion,
            float(n_rondas + 1)  # próxima ronda
        ], dtype=float)

        # Verificar que no hay NaN o Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la próxima jugada del oponente.

        Returns:
            Jugada predicha del oponente
        """
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()
        features_scaled = self.scaler.transform([features])
        prediccion = self.modelo.predict(features_scaled)[0]

        return NUM_A_JUGADA[int(prediccion)]

    def decidir_jugada(self) -> str:
        """
        Decide qué jugada hacer para ganar al oponente.
        Con aleatoriedad estratégica para no ser predecible.

        Returns:
            La jugada que gana a la predicción (80%) o aleatoria (20%)
        """
        # Primeras 3 rondas: aleatorio puro (modelo no tiene suficiente info)
        if len(self.historial) < 3:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Obtener predicción del modelo
        prediccion_oponente = self.predecir_jugada_oponente()

        # 80% del tiempo: jugar lo que le gana a la predicción
        # 20% del tiempo: jugar algo aleatorio (para evitar patrones)
        if np.random.random() < 0.80:
            jugada_ganadora = PIERDE_CONTRA[prediccion_oponente]
        else:
            # Aleatoriedad estratégica: evitar la jugada que el modelo predice demasiado
            opciones = ["piedra", "papel", "tijera"]
            jugada_ganadora = np.random.choice(opciones)

        return jugada_ganadora

    def diagnosticar_modelo(df: pd.DataFrame, modelo, scaler, X_test, y_test):
        """
        Diagnostica problemas de desbalanceo en el modelo.
        """
        print("\n" + "=" * 50)
        print("   DIAGNÓSTICO DEL MODELO")
        print("=" * 50)

        # 1. DISTRIBUCIÓN EN EL DATASET ORIGINAL
        print("\n📊 1. DISTRIBUCIÓN DE JUGADAS EN EL DATASET:")
        print("-" * 50)
        distribucion = df['jugada_j2'].value_counts()
        total = len(df)

        distribuciones = {}
        for jugada in ['piedra', 'papel', 'tijera']:
            count = distribucion.get(jugada, 0)
            porcentaje = (count / total) * 100
            distribuciones[jugada] = porcentaje
            barra = "█" * int(porcentaje / 2)
            print(f"  {jugada.capitalize():8} | {barra} {count:4} ({porcentaje:.1f}%)")

        # 2. PREDICCIONES DEL MODELO
        print("\n🤖 2. PREDICCIONES DEL MODELO EN TEST:")
        print("-" * 50)

        X_test_scaled = scaler.transform(X_test)
        y_pred = modelo.predict(X_test_scaled)

        pred_counts = pd.Series(y_pred).value_counts()
        total_pred = len(y_pred)

        predicciones = {}
        print("  El modelo predice:")
        for i, jugada in enumerate(['piedra', 'papel', 'tijera']):
            count = pred_counts.get(float(i), 0)
            porcentaje = (count / total_pred) * 100
            predicciones[jugada] = porcentaje
            barra = "█" * int(porcentaje / 2)

            # Detectar sesgo
            desviacion = abs(porcentaje - 33.33)
            if desviacion > 15:
                alerta = " ⚠ SESGO ALTO"
            elif desviacion > 8:
                alerta = " ⚠ Sesgo moderado"
            else:
                alerta = ""

            print(f"    {jugada.capitalize():8} | {barra} {count:4} ({porcentaje:.1f}%){alerta}")

        # 3. COMPARACIÓN
        print("\n📈 3. COMPARACIÓN DATASET vs MODELO:")
        print("-" * 50)
        print(f"  {'Jugada':10} | {'Dataset':>8} | {'Modelo':>8} | {'Diferencia':>11}")
        print("  " + "-" * 48)

        for jugada in ['piedra', 'papel', 'tijera']:
            dataset_pct = distribuciones[jugada]
            modelo_pct = predicciones[jugada]
            diff = modelo_pct - dataset_pct

            indicador = "↑" if diff > 3 else ("↓" if diff < -3 else "≈")
            print(f"  {jugada.capitalize():10} | {dataset_pct:7.1f}% | {modelo_pct:7.1f}% | {diff:+9.1f}% {indicador}")

        # 4. MATRIZ DE CONFUSIÓN
        print("\n📊 4. MATRIZ DE CONFUSIÓN:")
        print("-" * 50)
        cm = confusion_matrix(y_test, y_pred)

        print("\n           PREDICCIÓN")
        print("          Piedra  Papel  Tijera")
        labels = ['Piedra', 'Papel', 'Tijera']

        for i, label in enumerate(labels):
            print(f"  {label:7} ", end="")
            for j in range(3):
                valor = cm[i][j]
                if i == j:
                    print(f"  {valor:3} ✓ ", end="")
                else:
                    print(f"  {valor:3}   ", end="")
            print(f" | {cm[i].sum()}")

        # 5. DIAGNÓSTICO FINAL
        print("\n💡 DIAGNÓSTICO:")
        print("-" * 50)

        problemas = []
        for jugada in ['piedra', 'papel', 'tijera']:
            desv = abs(predicciones[jugada] - 33.33)
            if desv > 15:
                problemas.append((jugada, predicciones[jugada], "ALTO"))
            elif desv > 8:
                problemas.append((jugada, predicciones[jugada], "moderado"))

        if problemas:
            print("  ⚠️  PROBLEMAS DETECTADOS:\n")
            for jugada, pct, nivel in problemas:
                print(f"  - Sesgo {nivel} hacia '{jugada.upper()}' ({pct:.1f}%)")

            print("\n  🔧 SOLUCIONES RECOMENDADAS:\n")
            print("  1. Añadir class_weight='balanced' en los modelos")
            print("  2. Verificar balance del dataset de entrenamiento")
            print("  3. Implementar aleatoriedad estratégica (80/20)")
        else:
            print("  ✅ El modelo está razonablemente balanceado")

        print("\n" + "=" * 50)


def diagnosticar_modelo(df: pd.DataFrame, modelo, scaler, X_test, y_test):
    """
    Diagnostica problemas de desbalanceo en el modelo.
    """
    print("\n" + "=" * 50)
    print("   DIAGNÓSTICO DEL MODELO")
    print("=" * 50)

    # 1. DISTRIBUCIÓN EN EL DATASET ORIGINAL
    print("\n📊 1. DISTRIBUCIÓN DE JUGADAS EN EL DATASET:")
    print("-" * 50)
    distribucion = df['jugada_j2'].value_counts()
    total = len(df)

    distribuciones = {}
    for jugada in ['piedra', 'papel', 'tijera']:
        count = distribucion.get(jugada, 0)
        porcentaje = (count / total) * 100
        distribuciones[jugada] = porcentaje
        barra = "█" * int(porcentaje / 2)
        print(f"  {jugada.capitalize():8} | {barra} {count:4} ({porcentaje:.1f}%)")

    # 2. PREDICCIONES DEL MODELO
    print("\n🤖 2. PREDICCIONES DEL MODELO EN TEST:")
    print("-" * 50)

    X_test_scaled = scaler.transform(X_test)
    y_pred = modelo.predict(X_test_scaled)

    pred_counts = pd.Series(y_pred).value_counts()
    total_pred = len(y_pred)

    predicciones = {}
    print("  El modelo predice:")
    for i, jugada in enumerate(['piedra', 'papel', 'tijera']):
        count = pred_counts.get(float(i), 0)
        porcentaje = (count / total_pred) * 100
        predicciones[jugada] = porcentaje
        barra = "█" * int(porcentaje / 2)

        # Detectar sesgo
        desviacion = abs(porcentaje - 33.33)
        if desviacion > 15:
            alerta = " ⚠ SESGO ALTO"
        elif desviacion > 8:
            alerta = " ⚠ Sesgo moderado"
        else:
            alerta = ""

        print(f"    {jugada.capitalize():8} | {barra} {count:4} ({porcentaje:.1f}%){alerta}")

    # 3. COMPARACIÓN
    print("\n📈 3. COMPARACIÓN DATASET vs MODELO:")
    print("-" * 50)
    print(f"  {'Jugada':10} | {'Dataset':>8} | {'Modelo':>8} | {'Diferencia':>11}")
    print("  " + "-" * 48)

    for jugada in ['piedra', 'papel', 'tijera']:
        dataset_pct = distribuciones[jugada]
        modelo_pct = predicciones[jugada]
        diff = modelo_pct - dataset_pct

        indicador = "↑" if diff > 3 else ("↓" if diff < -3 else "≈")
        print(f"  {jugada.capitalize():10} | {dataset_pct:7.1f}% | {modelo_pct:7.1f}% | {diff:+9.1f}% {indicador}")

    # 4. MATRIZ DE CONFUSIÓN
    print("\n📊 4. MATRIZ DE CONFUSIÓN:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)

    print("\n           PREDICCIÓN")
    print("          Piedra  Papel  Tijera")
    labels = ['Piedra', 'Papel', 'Tijera']

    for i, label in enumerate(labels):
        print(f"  {label:7} ", end="")
        for j in range(3):
            valor = cm[i][j]
            if i == j:
                print(f"  {valor:3} ✓ ", end="")
            else:
                print(f"  {valor:3}   ", end="")
        print(f" | {cm[i].sum()}")

    # 5. DIAGNÓSTICO FINAL
    print("\n💡 DIAGNÓSTICO:")
    print("-" * 50)

    problemas = []
    for jugada in ['piedra', 'papel', 'tijera']:
        desv = abs(predicciones[jugada] - 33.33)
        if desv > 15:
            problemas.append((jugada, predicciones[jugada], "ALTO"))
        elif desv > 8:
            problemas.append((jugada, predicciones[jugada], "moderado"))

    if problemas:
        print("  ⚠️  PROBLEMAS DETECTADOS:\n")
        for jugada, pct, nivel in problemas:
            print(f"  - Sesgo {nivel} hacia '{jugada.upper()}' ({pct:.1f}%)")

        print("\n  🔧 SOLUCIONES RECOMENDADAS:\n")
        print("  1. Añadir class_weight='balanced' en los modelos")
        print("  2. Verificar balance del dataset de entrenamiento")
        print("  3. Implementar aleatoriedad estratégica (80/20)")
    else:
        print("  ✅ El modelo está razonablemente balanceado")

    print("\n" + "=" * 50)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para entrenar el modelo."""
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    try:
        # 1. Cargar datos
        print("\n[1/6] Cargando datos...")
        df = cargar_datos()

        # AÑADIR ESTE BLOQUE:
        # Verificar balance del dataset
        print("\n📊 Balance del dataset original:")
        distribucion = df['jugada_j2'].value_counts()
        total = len(df)
        desbalanceado = False

        for jugada in ['piedra', 'papel', 'tijera']:
            count = distribucion.get(jugada, 0)
            pct = (count / total) * 100
            desv = abs(pct - 33.33)

            if desv > 10:
                print(f"  ⚠️  {jugada.capitalize()}: {count} ({pct:.1f}%) - DESBALANCEADO")
                desbalanceado = True
            else:
                print(f"  ✓ {jugada.capitalize()}: {count} ({pct:.1f}%)")

        if desbalanceado:
            print("\n⚠️  Dataset desbalanceado detectado.")
            print("   El modelo usará class_weight='balanced' para compensar.")

        # 2. Preparar datos
        print("\n[2/6] Preparando datos...")
        df = preparar_datos(df)

        # 3. Crear features
        print("\n[3/6] Creando features...")
        df = crear_features(df)

        # 4. Seleccionar features
        print("\n[4/6] Seleccionando features...")
        X, y = seleccionar_features(df)

        # 5. Entrenar modelo
        print("\n[5/7] Entrenando modelos...")
        modelo, scaler = entrenar_modelo(X, y)

        # 5.5 DIAGNÓSTICO DEL MODELO (NUEVO)
        print("\n[6/7] Diagnosticando modelo...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        diagnosticar_modelo(df, modelo, scaler, X_test, y_test)

        # 6. Guardar modelo
        print("\n[7/7] Guardando modelo...")
        guardar_modelo(modelo, scaler)

        print("\n" + "="*50)
        print("   ✓ ENTRENAMIENTO COMPLETADO")
        print("="*50)
        print("\nAhora puedes usar tu modelo en el juego!")
        print("Ejecuta: python src/juego.py")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nAsegúrate de tener el archivo data/partidas.csv")
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()