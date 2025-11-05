"""
Feature Engineering para Piedra, Papel o Tijera
Ejemplos prácticos de implementación

Clase 5: Fundamentos de IA

IMPORTANTE: Esta clase solo requiere los datos básicos del CSV:
- jugada_jugador
- jugada_oponente
- numero_ronda

Los resultados y todas las features se calculan automáticamente.
NO necesitas tiempo de reacción ni ningún otro dato adicional.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


class PPTFeatureEngineering:
    """
    Clase completa para Feature Engineering del juego Piedra, Papel o Tijera
    """

    def __init__(self):
        self.jugadas_map = {"piedra": 0, "papel": 1, "tijera": 2}
        self.jugadas_reverse = {0: "piedra", 1: "papel", 2: "tijera"}
        self.counter_map = {
            "piedra": "papel",
            "papel": "tijera",
            "tijera": "piedra"
        }

    def calcular_resultado(self, jugada_jugador, jugada_oponente):
        """
        Calcula el resultado desde la perspectiva del jugador

        Args:
            jugada_jugador: jugada del jugador
            jugada_oponente: jugada del oponente

        Returns:
            str: "victoria", "derrota" o "empate"
        """
        if jugada_jugador == jugada_oponente:
            return "empate"

        gana = {
            "piedra": "tijera",
            "papel": "piedra",
            "tijera": "papel"
        }

        if gana[jugada_jugador] == jugada_oponente:
            return "victoria"
        else:
            return "derrota"

    # ========== FEATURES DE FRECUENCIA ==========

    def calcular_frecuencias(self, jugadas, ventana=None):
        """
        Calcula la frecuencia de cada jugada

        Args:
            jugadas: lista de jugadas ["piedra", "papel", ...]
            ventana: número de últimas jugadas a considerar (None = todas)

        Returns:
            dict con frecuencias {jugada: frecuencia}
        """
        if ventana:
            jugadas = jugadas[-ventana:]

        total = len(jugadas)
        if total == 0:
            return {"piedra": 0.0, "papel": 0.0, "tijera": 0.0}

        conteo = Counter(jugadas)
        return {
            "piedra": conteo.get("piedra", 0) / total,
            "papel": conteo.get("papel", 0) / total,
            "tijera": conteo.get("tijera", 0) / total
        }

    def features_frecuencia(self, jugadas):
        """Genera features de frecuencia (global y ventanas)"""
        features = {}

        # Frecuencias globales
        freq_global = self.calcular_frecuencias(jugadas)
        features['freq_global_piedra'] = freq_global['piedra']
        features['freq_global_papel'] = freq_global['papel']
        features['freq_global_tijera'] = freq_global['tijera']

        # Frecuencias en ventanas
        for ventana in [3, 5, 10]:
            freq = self.calcular_frecuencias(jugadas, ventana=ventana)
            features[f'freq_{ventana}_piedra'] = freq['piedra']
            features[f'freq_{ventana}_papel'] = freq['papel']
            features[f'freq_{ventana}_tijera'] = freq['tijera']

        # Cambios de tendencia
        freq_reciente = self.calcular_frecuencias(jugadas, ventana=5)
        features['cambio_tend_piedra'] = freq_reciente['piedra'] - freq_global['piedra']
        features['cambio_tend_papel'] = freq_reciente['papel'] - freq_global['papel']
        features['cambio_tend_tijera'] = freq_reciente['tijera'] - freq_global['tijera']

        # Jugada más común
        if jugadas:
            jugada_comun = Counter(jugadas).most_common(1)[0][0]
            features['jugada_mas_comun_piedra'] = 1 if jugada_comun == "piedra" else 0
            features['jugada_mas_comun_papel'] = 1 if jugada_comun == "papel" else 0
            features['jugada_mas_comun_tijera'] = 1 if jugada_comun == "tijera" else 0

        return features

    # ========== FEATURES DE PATRONES SECUENCIALES ==========

    def crear_lag_features(self, jugadas, n_lags=3):
        """
        Crea features de jugadas anteriores (lags)

        Args:
            jugadas: lista de jugadas
            n_lags: número de lags a crear

        Returns:
            dict con features de lag (one-hot encoded)
        """
        features = {}

        for i in range(1, n_lags + 1):
            if len(jugadas) >= i:
                jugada = jugadas[-i]
                # One-hot encoding
                features[f'lag_{i}_piedra'] = 1 if jugada == "piedra" else 0
                features[f'lag_{i}_papel'] = 1 if jugada == "papel" else 0
                features[f'lag_{i}_tijera'] = 1 if jugada == "tijera" else 0
            else:
                # No hay suficientes datos
                features[f'lag_{i}_piedra'] = 0
                features[f'lag_{i}_papel'] = 0
                features[f'lag_{i}_tijera'] = 0

        return features

    def crear_patron_features(self, jugadas, n=2):
        """
        Crea features basadas en n-gramas (patrones de n jugadas consecutivas)

        Args:
            jugadas: lista de jugadas
            n: tamaño del patrón (2=bigramas, 3=trigramas)

        Returns:
            dict con features de patrones
        """
        features = {}

        if len(jugadas) < n:
            features[f'patron_{n}_nuevo'] = 1
            features[f'patron_{n}_frecuencia'] = 0
            return features

        # Patrón actual (últimas n jugadas)
        patron_actual = tuple(jugadas[-n:])

        # Contar ocurrencias de este patrón
        count = 0
        for i in range(len(jugadas) - n + 1):
            if tuple(jugadas[i:i+n]) == patron_actual:
                count += 1

        total_patrones = max(1, len(jugadas) - n + 1)
        features[f'patron_{n}_frecuencia'] = count / total_patrones
        features[f'patron_{n}_nuevo'] = 1 if count == 1 else 0

        # Representación del patrón (one-hot encoding simplificado)
        patron_str = "".join([j[0] for j in patron_actual])  # "pp" para piedra-papel
        features[f'patron_{n}_string'] = patron_str

        return features

    def features_patrones(self, jugadas):
        """Genera todas las features de patrones"""
        features = {}

        # Lags
        lag_feats = self.crear_lag_features(jugadas, n_lags=3)
        features.update(lag_feats)

        # Bigramas y trigramas
        for n in [2, 3]:
            patron_feats = self.crear_patron_features(jugadas, n=n)
            features.update(patron_feats)

        # Cambios consecutivos
        if len(jugadas) >= 2:
            cambios = sum(1 for i in range(1, min(5, len(jugadas)))
                         if jugadas[-i] != jugadas[-i-1])
            features['cambios_ultimos_5'] = cambios
            features['repeticiones_consecutivas'] = 1 if jugadas[-1] == jugadas[-2] else 0
        else:
            features['cambios_ultimos_5'] = 0
            features['repeticiones_consecutivas'] = 0

        return features

    # ========== FEATURES DE RACHAS ==========

    def calcular_rachas(self, resultados):
        """
        Calcula rachas de victorias/derrotas/empates

        Args:
            resultados: lista de resultados ["victoria", "derrota", ...]

        Returns:
            dict con features de rachas
        """
        features = {
            'racha_victorias': 0,
            'racha_derrotas': 0,
            'racha_empates': 0,
            'max_racha_victorias': 0,
            'max_racha_derrotas': 0,
            'tipo_racha': 'ninguna'
        }

        if not resultados:
            return features

        # Racha actual
        racha_actual = 1
        ultimo_resultado = resultados[-1]

        for i in range(len(resultados) - 2, -1, -1):
            if resultados[i] == ultimo_resultado:
                racha_actual += 1
            else:
                break

        if ultimo_resultado == "victoria":
            features['racha_victorias'] = racha_actual
            features['tipo_racha'] = 'victoria'
        elif ultimo_resultado == "derrota":
            features['racha_derrotas'] = racha_actual
            features['tipo_racha'] = 'derrota'
        elif ultimo_resultado == "empate":
            features['racha_empates'] = racha_actual
            features['tipo_racha'] = 'empate'

        # Rachas máximas
        racha_temp = 1
        for i in range(1, len(resultados)):
            if resultados[i] == resultados[i-1]:
                racha_temp += 1
            else:
                if resultados[i-1] == "victoria":
                    features['max_racha_victorias'] = max(features['max_racha_victorias'], racha_temp)
                elif resultados[i-1] == "derrota":
                    features['max_racha_derrotas'] = max(features['max_racha_derrotas'], racha_temp)
                racha_temp = 1

        # Última racha
        if resultados[-1] == "victoria":
            features['max_racha_victorias'] = max(features['max_racha_victorias'], racha_temp)
        elif resultados[-1] == "derrota":
            features['max_racha_derrotas'] = max(features['max_racha_derrotas'], racha_temp)

        # Proximidad a récord
        features['cerca_record_victorias'] = 1 if features['racha_victorias'] >= features['max_racha_victorias'] - 1 else 0

        return features

    def features_rachas(self, resultados):
        """Genera todas las features de rachas"""
        return self.calcular_rachas(resultados)

    # ========== FEATURES TEMPORALES ==========

    def features_temporales(self, numero_ronda, total_rondas_esperadas=50):
        """
        Genera features temporales (fase del juego)

        Args:
            numero_ronda: ronda actual
            total_rondas_esperadas: total esperado de rondas

        Returns:
            dict con features temporales
        """
        features = {}

        # Posición en el juego
        features['numero_ronda'] = numero_ronda
        features['progreso_juego'] = numero_ronda / total_rondas_esperadas

        # Fase del juego
        progreso = features['progreso_juego']
        if progreso < 0.33:
            features['fase_inicio'] = 1
            features['fase_medio'] = 0
            features['fase_final'] = 0
        elif progreso < 0.66:
            features['fase_inicio'] = 0
            features['fase_medio'] = 1
            features['fase_final'] = 0
        else:
            features['fase_inicio'] = 0
            features['fase_medio'] = 0
            features['fase_final'] = 1

        return features

    # ========== FEATURES DE ENTROPÍA ==========

    def calcular_entropia(self, jugadas):
        """
        Calcula la entropía de Shannon

        Entropía = -Σ P(x) * log2(P(x))

        Args:
            jugadas: lista de jugadas

        Returns:
            float: entropía (0 = muy predecible, ~1.58 = muy aleatorio)
        """
        if not jugadas:
            return 0.0

        freq = self.calcular_frecuencias(jugadas)
        probs = [p for p in freq.values() if p > 0]

        if not probs:
            return 0.0

        return entropy(probs, base=2)

    def features_entropia(self, jugadas):
        """Genera features de entropía"""
        features = {}

        # Entropía global
        features['entropia_global'] = self.calcular_entropia(jugadas)

        # Entropía en ventanas
        for ventana in [3, 5, 10]:
            if len(jugadas) >= ventana:
                features[f'entropia_{ventana}'] = self.calcular_entropia(jugadas[-ventana:])
            else:
                features[f'entropia_{ventana}'] = 0

        # Cambio en entropía
        if len(jugadas) >= 5:
            cambio = features['entropia_5'] - features['entropia_global']
            features['cambio_entropia'] = cambio
            features['volviendose_mas_aleatorio'] = 1 if cambio > 0.2 else 0
        else:
            features['cambio_entropia'] = 0
            features['volviendose_mas_aleatorio'] = 0

        # Nivel de predictibilidad
        H = features['entropia_global']
        if H < 0.5:
            nivel = 0  # muy predecible
        elif H < 1.0:
            nivel = 1  # algo predecible
        elif H < 1.4:
            nivel = 2  # poco predecible
        else:
            nivel = 3  # casi aleatorio

        features['nivel_predictibilidad'] = nivel

        return features

    # ========== FEATURES DE TRANSICIÓN (MARKOV) ==========

    def calcular_matriz_transicion(self, jugadas):
        """
        Calcula la matriz de transición P(siguiente | actual)

        Returns:
            dict: {(jugada_actual, jugada_siguiente): probabilidad}
        """
        if len(jugadas) < 2:
            return {}

        # Contar transiciones
        transiciones = defaultdict(int)
        conteos_desde = defaultdict(int)

        for i in range(len(jugadas) - 1):
            actual = jugadas[i]
            siguiente = jugadas[i + 1]
            transiciones[(actual, siguiente)] += 1
            conteos_desde[actual] += 1

        # Calcular probabilidades
        matriz = {}
        for (actual, siguiente), count in transiciones.items():
            if conteos_desde[actual] > 0:
                matriz[(actual, siguiente)] = count / conteos_desde[actual]

        return matriz

    def features_markov(self, jugadas):
        """Genera features basadas en cadenas de Markov"""
        features = {}

        matriz = self.calcular_matriz_transicion(jugadas)

        if not matriz:
            # Sin suficientes datos
            for jugada_actual in ["piedra", "papel", "tijera"]:
                for jugada_sig in ["piedra", "papel", "tijera"]:
                    features[f'markov_{jugada_actual}_{jugada_sig}'] = 0.33
            features['markov_confianza'] = 0
            return features

        # Probabilidades de transición desde cada jugada
        for jugada_actual in ["piedra", "papel", "tijera"]:
            for jugada_sig in ["piedra", "papel", "tijera"]:
                prob = matriz.get((jugada_actual, jugada_sig), 0.0)
                features[f'markov_{jugada_actual}_{jugada_sig}'] = prob

        # Predicción basada en última jugada
        if jugadas:
            ultima = jugadas[-1]
            transiciones_desde_ultima = {
                sig: matriz.get((ultima, sig), 0.0)
                for sig in ["piedra", "papel", "tijera"]
            }
            if transiciones_desde_ultima:
                prediccion = max(transiciones_desde_ultima, key=transiciones_desde_ultima.get)
                confianza = transiciones_desde_ultima[prediccion]
                features['markov_confianza'] = confianza
                features['markov_pred_piedra'] = 1 if prediccion == "piedra" else 0
                features['markov_pred_papel'] = 1 if prediccion == "papel" else 0
                features['markov_pred_tijera'] = 1 if prediccion == "tijera" else 0

        return features

    # ========== FEATURES DE RESPUESTA A RESULTADOS ==========

    def features_respuesta_resultados(self, jugadas, resultados):
        """Genera features sobre cómo responde el oponente a victorias/derrotas"""
        features = {}

        if len(jugadas) < 2 or len(resultados) < 2:
            return {
                'cambia_despues_ganar': 0,
                'cambia_despues_perder': 0,
                'ultima_fue_victoria': 0
            }

        # Última victoria/derrota
        features['ultima_fue_victoria'] = 1 if resultados[-1] == "victoria" else 0
        features['ultima_fue_derrota'] = 1 if resultados[-1] == "derrota" else 0

        # Frecuencia de cambiar después de ganar/perder
        cambios_despues_ganar = 0
        victorias = 0
        cambios_despues_perder = 0
        derrotas = 0

        for i in range(len(resultados) - 1):
            if resultados[i] == "victoria":
                victorias += 1
                if jugadas[i] != jugadas[i + 1]:
                    cambios_despues_ganar += 1
            elif resultados[i] == "derrota":
                derrotas += 1
                if jugadas[i] != jugadas[i + 1]:
                    cambios_despues_perder += 1

        features['cambia_despues_ganar'] = cambios_despues_ganar / max(1, victorias)
        features['cambia_despues_perder'] = cambios_despues_perder / max(1, derrotas)

        return features

    # ========== GENERADOR COMPLETO ==========

    def generar_features_completas(self, jugadas_oponente, jugadas_jugador,
                                   numero_ronda, total_rondas=50):
        """
        Genera el vector completo de features para una ronda

        IMPORTANTE: Solo requiere los datos básicos del CSV:
        - jugadas_oponente
        - jugadas_jugador
        - numero_ronda

        Los resultados se calculan automáticamente.

        Args:
            jugadas_oponente: lista de jugadas del oponente
            jugadas_jugador: lista de jugadas del jugador
            numero_ronda: número de ronda actual
            total_rondas: total estimado de rondas (default: 50)

        Returns:
            dict con todas las features
        """
        features = {}

        # Calcular resultados a partir de las jugadas
        resultados = [self.calcular_resultado(j_jug, j_op)
                     for j_jug, j_op in zip(jugadas_jugador, jugadas_oponente)]

        # 1. Features de frecuencia
        freq_features = self.features_frecuencia(jugadas_oponente)
        features.update(freq_features)

        # 2. Features de patrones
        patron_features = self.features_patrones(jugadas_oponente)
        features.update(patron_features)

        # 3. Features de rachas
        racha_features = self.features_rachas(resultados)
        features.update(racha_features)

        # 4. Features temporales (solo fase del juego)
        temp_features = self.features_temporales(numero_ronda, total_rondas)
        features.update(temp_features)

        # 5. Features de entropía
        entropia_features = self.features_entropia(jugadas_oponente)
        features.update(entropia_features)

        # 6. Features de Markov
        markov_features = self.features_markov(jugadas_oponente)
        features.update(markov_features)

        # 7. Features de respuesta a resultados
        respuesta_features = self.features_respuesta_resultados(jugadas_oponente, resultados)
        features.update(respuesta_features)

        return features


# ========== FUNCIONES DE DEMOSTRACIÓN ==========

def ejemplo_basico():
    """Ejemplo básico de uso"""
    print("=" * 60)
    print("EJEMPLO BÁSICO: Generación de Features")
    print("=" * 60)

    fe = PPTFeatureEngineering()

    # Datos básicos (lo que tienen los alumnos en su CSV)
    jugadas_jugador = [
        "papel", "piedra", "piedra", "papel", "piedra",
        "piedra", "tijera", "tijera", "papel", "tijera"
    ]
    jugadas_oponente = [
        "piedra", "piedra", "papel", "piedra", "tijera",
        "piedra", "papel", "papel", "piedra", "papel"
    ]

    print("\nDatos básicos del CSV:")
    print(f"  Jugadas del jugador: {len(jugadas_jugador)} rondas")
    print(f"  Jugadas del oponente: {len(jugadas_oponente)} rondas")
    print(f"  Ronda actual: 10")

    # Generar features (los resultados se calculan automáticamente)
    features = fe.generar_features_completas(
        jugadas_oponente=jugadas_oponente,
        jugadas_jugador=jugadas_jugador,
        numero_ronda=10,
        total_rondas=50
    )

    # Mostrar features
    print(f"\nTotal de features generadas: {len(features)}\n")

    # Agrupar por categoría
    categorias = {
        'Frecuencia': [k for k in features if 'freq' in k or 'cambio_tend' in k],
        'Patrones': [k for k in features if 'lag' in k or 'patron' in k or 'cambio' in k],
        'Rachas': [k for k in features if 'racha' in k],
        'Temporales': [k for k in features if 'tiempo' in k or 'fase' in k or 'ronda' in k],
        'Entropía': [k for k in features if 'entropia' in k or 'predictibilidad' in k],
        'Markov': [k for k in features if 'markov' in k],
        'Respuesta': [k for k in features if 'despues' in k or 'ultima_fue' in k]
    }

    for categoria, keys in categorias.items():
        print(f"\n{'=' * 60}")
        print(f"{categoria.upper()}")
        print('=' * 60)
        for key in keys:
            if key in features:
                valor = features[key]
                if isinstance(valor, float):
                    print(f"  {key:35s}: {valor:.4f}")
                else:
                    print(f"  {key:35s}: {valor}")


def ejemplo_evolucion_features():
    """Muestra cómo evolucionan las features durante el juego"""
    print("\n" + "=" * 60)
    print("EJEMPLO: Evolución de Features Durante el Juego")
    print("=" * 60)

    fe = PPTFeatureEngineering()

    # Simular un juego
    jugadas_oponente = []
    resultados = []

    # Primeras 5 rondas: oponente juega piedra frecuentemente
    for _ in range(5):
        jugadas_oponente.append("piedra")
        resultados.append("victoria")

    # Siguientes 5 rondas: cambia a papel
    for _ in range(5):
        jugadas_oponente.append("papel")
        resultados.append("derrota")

    # Últimas 5 rondas: mezcla
    jugadas_oponente.extend(["tijera", "piedra", "papel", "tijera", "papel"])
    resultados.extend(["victoria", "empate", "derrota", "victoria", "derrota"])

    # Analizar en diferentes momentos
    momentos = [5, 10, 15]

    print("\nAnálisis de features clave en diferentes momentos:\n")

    for momento in momentos:
        jugadas_hasta_ahora = jugadas_oponente[:momento]
        resultados_hasta_ahora = resultados[:momento]

        freq = fe.calcular_frecuencias(jugadas_hasta_ahora)
        entropia = fe.calcular_entropia(jugadas_hasta_ahora)

        print(f"Ronda {momento}:")
        print(f"  Jugadas: {jugadas_hasta_ahora[-5:]}")
        print(f"  Frecuencia piedra: {freq['piedra']:.2f}")
        print(f"  Frecuencia papel:  {freq['papel']:.2f}")
        print(f"  Frecuencia tijera: {freq['tijera']:.2f}")
        print(f"  Entropía:          {entropia:.2f} ({'predecible' if entropia < 1.0 else 'aleatorio'})")
        print()


def ejemplo_visualizacion():
    """Visualiza la importancia de diferentes features"""
    print("\n" + "=" * 60)
    print("EJEMPLO: Visualización de Features")
    print("=" * 60)

    fe = PPTFeatureEngineering()

    # Generar un dataset más grande
    np.random.seed(42)
    n_rondas = 50

    # Simular oponente con patrón: prefiere piedra 50%, luego papel 30%, tijera 20%
    jugadas = np.random.choice(["piedra", "papel", "tijera"],
                               size=n_rondas,
                               p=[0.5, 0.3, 0.2])

    # Calcular features para cada ronda
    frecuencias_piedra = []
    entropias = []

    for i in range(5, n_rondas):
        jugadas_hasta_ahora = jugadas[:i].tolist()
        freq = fe.calcular_frecuencias(jugadas_hasta_ahora)
        entropia = fe.calcular_entropia(jugadas_hasta_ahora)

        frecuencias_piedra.append(freq['piedra'])
        entropias.append(entropia)

    # Visualizar
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfico 1: Evolución de frecuencia de "piedra"
    axes[0].plot(range(5, n_rondas), frecuencias_piedra, 'b-', linewidth=2)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Frecuencia real (50%)')
    axes[0].set_xlabel('Número de Ronda')
    axes[0].set_ylabel('Frecuencia de "Piedra"')
    axes[0].set_title('Evolución de la Frecuencia de "Piedra"')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico 2: Evolución de entropía
    axes[1].plot(range(5, n_rondas), entropias, 'g-', linewidth=2)
    axes[1].axhline(y=1.58, color='r', linestyle='--', label='Máxima entropía (aleatorio)')
    axes[1].set_xlabel('Número de Ronda')
    axes[1].set_ylabel('Entropía')
    axes[1].set_title('Evolución de la Entropía (Aleatoriedad)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolucion_features_ppt.png', dpi=100)
    print("\nGráfico guardado como 'evolucion_features_ppt.png'")


# ========== EJECUCIÓN PRINCIPAL ==========

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 10 + "FEATURE ENGINEERING PARA PIEDRA, PAPEL O TIJERA")
    print("=" * 70)

    # Ejecutar ejemplos
    ejemplo_basico()
    ejemplo_evolucion_features()

    try:
        ejemplo_visualizacion()
    except Exception as e:
        print(f"\nNota: No se pudo generar visualización: {e}")
        print("Asegúrate de tener matplotlib instalado: pip install matplotlib")

    print("\n" + "=" * 70)
    print("Ejemplos completados. Revisa el código para ver implementaciones.")
    print("=" * 70)
