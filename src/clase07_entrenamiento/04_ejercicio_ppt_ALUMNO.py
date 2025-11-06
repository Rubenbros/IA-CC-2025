"""
EJERCICIO PARA ALUMNOS: Entrenar Modelo para Piedra, Papel o Tijera
====================================================================

OBJETIVO:
Aplicar TODO lo aprendido en esta clase para crear un modelo que prediga
la próxima jugada del oponente en PPT usando tus DATOS REALES de partidas.

REQUISITOS PREVIOS:
- Debes tener un CSV con datos de tus partidas contra compañeros
- El CSV debe tener al menos estas columnas:
  * jugada_jugador (tu jugada: piedra/papel/tijera)
  * jugada_oponente (jugada del oponente: piedra/papel/tijera)
  * numero_ronda (número de ronda: 1, 2, 3, ...)

TAREAS:
1. Usar tus datos reales de partidas (mínimo 100 rondas)
2. Crear función para generar features básicas
3. Preparar datos con train/test split
4. Entrenar múltiples modelos (KNN, Decision Tree, Random Forest)
5. Comparar resultados
6. Implementar un jugador IA que use el mejor modelo

OBJETIVO:
- Lograr >50% accuracy (mejor que aleatorio 33%)
- Comparar al menos 3 modelos diferentes
- Mostrar confusion matrix
- (Bonus) Implementar clase JugadorIA

¡BUENA SUERTE!
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# PARTE 1: GENERAR FEATURES
# =============================================================================

class PPTFeatureGenerator:
    """
    Genera features para el modelo de PPT

    TODO: Implementa los métodos para generar features básicas
    """

    def __init__(self):
        self.jugadas_counter = {
            "piedra": "papel",
            "papel": "tijera",
            "tijera": "piedra"
        }

    def calcular_resultado(self, jugada_jugador, jugada_oponente):
        """
        Calcula el resultado de una ronda

        TODO: Implementa esta función
        Debe retornar: "victoria", "derrota", o "empate"
        """
        # TU CÓDIGO AQUÍ
        pass

    def generar_features_basicas(self, historial_oponente, historial_jugador, numero_ronda):
        """
        Genera features básicas para una ronda

        Args:
            historial_oponente: lista de jugadas del oponente hasta ahora
            historial_jugador: lista de jugadas del jugador hasta ahora
            numero_ronda: número de ronda actual

        Returns:
            dict con features

        TODO: Genera al menos estas features:
        - freq_piedra, freq_papel, freq_tijera (frecuencia global)
        - freq_5_piedra, freq_5_papel, freq_5_tijera (últimas 5 jugadas)
        - lag_1_piedra, lag_1_papel, lag_1_tijera (última jugada, one-hot)
        - lag_2_* (penúltima jugada)
        - racha_victorias, racha_derrotas
        - numero_ronda
        - fase_inicio, fase_medio, fase_final (one-hot)

        PISTA: Revisa los ejercicios de Feature Engineering (Clase 06)
        """
        features = {}

        # TODO: Implementa features de frecuencia global
        if historial_oponente:
            total = len(historial_oponente)
            # TU CÓDIGO AQUÍ: Calcula freq_piedra, freq_papel, freq_tijera
            pass
        else:
            features['freq_piedra'] = 0.33
            features['freq_papel'] = 0.33
            features['freq_tijera'] = 0.33

        # TODO: Implementa features de frecuencia reciente (últimas 5)
        # TU CÓDIGO AQUÍ

        # TODO: Implementa lag features (última y penúltima jugada)
        # TU CÓDIGO AQUÍ

        # TODO: Implementa features de rachas
        # TU CÓDIGO AQUÍ

        # TODO: Implementa features temporales
        features['numero_ronda'] = numero_ronda
        # TU CÓDIGO AQUÍ: fase_inicio, fase_medio, fase_final

        return features


# =============================================================================
# PARTE 2: PREPARAR DATOS
# =============================================================================

def cargar_y_preparar_datos(archivo_csv):
    """
    Carga datos de partidas y genera features

    Args:
        archivo_csv: Ruta a TU archivo CSV con datos de partidas
                     Debe tener columnas: jugada_jugador, jugada_oponente, numero_ronda

    TODO: Implementa esta función
    1. Cargar CSV con pandas
    2. Verificar que tiene las columnas necesarias
    3. Para cada ronda, generar features basadas en historial previo
    4. Target = jugada actual del oponente
    5. Retornar X (features), y (target)

    IMPORTANTE: Usa solo el historial HASTA cada ronda (no hacer trampa)
    """
    print("=" * 70)
    print("PASO 1: CARGAR Y PREPARAR DATOS")
    print("=" * 70)

    # TODO: Cargar TU CSV
    # TU CÓDIGO AQUÍ
    # df = pd.read_csv(archivo_csv)

    # TODO: Verificar columnas
    # columnas_necesarias = ['jugada_jugador', 'jugada_oponente', 'numero_ronda']
    # if not all(col in df.columns for col in columnas_necesarias):
    #     print(f"ERROR: El CSV debe tener columnas: {columnas_necesarias}")
    #     return None, None

    # TODO: Generar features para cada ronda
    # TU CÓDIGO AQUÍ
    # feature_gen = PPTFeatureGenerator()
    # lista_features = []
    # lista_targets = []
    # for idx in range(len(df)):
    #     ...

    # TODO: Crear DataFrames X e y
    # TU CÓDIGO AQUÍ
    # X = pd.DataFrame(lista_features)
    # y = pd.Series(lista_targets)

    # print(f"\n✓ Features generadas: {X.shape[0]} muestras, {X.shape[1]} features")

    # return X, y
    pass


# =============================================================================
# PARTE 3: ENTRENAR MODELOS
# =============================================================================

def entrenar_y_comparar_modelos(X_train, X_test, y_train, y_test):
    """
    Entrena múltiples modelos y los compara

    TODO: Implementa esta función
    1. Definir diccionario con al menos 3 modelos
    2. Entrenar cada modelo
    3. Evaluar en train y test
    4. Mostrar tabla comparativa
    5. Retornar el mejor modelo
    """
    print("\n" + "=" * 70)
    print("PASO 2: ENTRENAR Y COMPARAR MODELOS")
    print("=" * 70)

    # TODO: Define modelos a probar
    # TU CÓDIGO AQUÍ
    # modelos = {
    #     'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    #     'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    #     'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    # }

    # TODO: Entrena y evalúa cada modelo
    # TU CÓDIGO AQUÍ

    # TODO: Muestra resultados en tabla
    # TU CÓDIGO AQUÍ

    # TODO: Retorna el mejor modelo
    # TU CÓDIGO AQUÍ
    pass


# =============================================================================
# PARTE 4: EVALUACIÓN DETALLADA
# =============================================================================

def evaluar_mejor_modelo(nombre_modelo, modelo, X_test, y_test):
    """
    Evaluación detallada del mejor modelo

    TODO: Implementa esta función
    1. Hacer predicciones en test
    2. Calcular accuracy
    3. Mostrar confusion matrix
    4. Mostrar classification report
    5. (Bonus) Mostrar feature importance si es árbol/RF
    """
    print("\n" + "=" * 70)
    print(f"PASO 3: EVALUACIÓN DETALLADA - {nombre_modelo}")
    print("=" * 70)

    # TODO: Predicciones
    # TU CÓDIGO AQUÍ

    # TODO: Accuracy
    # TU CÓDIGO AQUÍ

    # TODO: Confusion Matrix
    # TU CÓDIGO AQUÍ

    # TODO: Classification Report
    # TU CÓDIGO AQUÍ

    pass


# =============================================================================
# PARTE 5: JUGADOR IA (BONUS)
# =============================================================================

class JugadorIA:
    """
    Jugador de PPT con IA

    TODO: Implementa esta clase
    """

    def __init__(self, modelo, feature_generator):
        self.modelo = modelo
        self.feature_gen = feature_generator
        self.historial_oponente = []
        self.historial_propio = []
        self.numero_ronda = 0

        self.counter = {
            'piedra': 'papel',
            'papel': 'tijera',
            'tijera': 'piedra'
        }

    def predecir_y_jugar(self):
        """
        Predice la próxima jugada del oponente y devuelve el counter

        TODO: Implementa esta función
        1. Incrementar número de ronda
        2. Si es ronda 1, jugar aleatorio
        3. Generar features basadas en historial
        4. Predecir jugada del oponente con el modelo
        5. Jugar el counter
        6. Retornar tu jugada

        Returns:
            str: jugada a realizar ('piedra', 'papel', o 'tijera')
        """
        self.numero_ronda += 1

        # TODO: Implementa la lógica
        # TU CÓDIGO AQUÍ
        pass

    def registrar_resultado(self, mi_jugada, jugada_oponente):
        """
        Registra el resultado de una ronda

        TODO: Implementa esta función
        Actualiza los historiales
        """
        # TU CÓDIGO AQUÍ
        pass


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Flujo completo del ejercicio

    TODO: Implementa el flujo completo:
    1. Especificar ruta a TU archivo CSV con datos reales
    2. Cargar y preparar datos
    3. Train/test split
    4. Entrenar y comparar modelos
    5. Evaluación detallada
    6. (Bonus) Simular partida con JugadorIA
    """
    print("\n" + "█" * 35)
    print("EJERCICIO: ENTRENAR MODELO PARA PPT")
    print("█" * 35)

    # TODO: Especifica la ruta a TU CSV con datos de partidas
    # Ejemplo: archivo_csv = 'mis_partidas_ppt.csv'
    # archivo_csv = '???'  # ← Cambia esto

    print("\n⚠️  IMPORTANTE: Debes tener tus datos de partidas en CSV")
    print("    Mínimo 100 rondas jugadas contra compañeros")
    print("    Columnas necesarias: jugada_jugador, jugada_oponente, numero_ronda")
    print()

    # TODO: Implementa el flujo completo aquí
    # TU CÓDIGO AQUÍ

    print("\n" + "█" * 35)
    print("¡EJERCICIO COMPLETADO!")
    print("█" * 35)


if __name__ == "__main__":
    # Descomenta cuando hayas implementado main()
    # main()

    print("=" * 70)
    print("INSTRUCCIONES")
    print("=" * 70)
    print("\n1. ANTES DE EMPEZAR:")
    print("   - Debes tener un CSV con tus partidas (mínimo 100 rondas)")
    print("   - Si no tienes datos, juega más partidas con compañeros")
    print("   - Formato CSV necesario:")
    print("     numero_ronda,jugada_jugador,jugada_oponente")
    print("     1,piedra,papel")
    print("     2,tijera,tijera")
    print("     ...")
    print("\n2. Implementa todas las funciones marcadas con TODO")
    print("\n3. Ejecuta este script:")
    print("   python 04_ejercicio_ppt_ALUMNO.py")
    print("\n4. OBJETIVO: Lograr >50% accuracy en test")
    print("\n" + "=" * 70)
    print("RECURSOS:")
    print("- Repasa ejemplos 01-03")
    print("- Revisa Clase 06 (Feature Engineering)")
    print("- Lee clases/07-entrenamiento-modelos/README.md")
    print("=" * 70)
