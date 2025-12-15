# src/modelo.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter, deque

# --- Configuración de rutas ---
partidas_csv = Path(__file__).parent.parent / "data" / "partidas.csv"
modelo_pkl = Path(__file__).parent.parent / "models" / "modelo_entrenado.pkl"

# --- Mapas de jugadas ---
JUGADAS = ["piedra", "papel", "tijera"]
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}


def codificar_jugada(jugada):
    return JUGADAS.index(jugada)


def decodificar_jugada(idx):
    return JUGADAS[idx]


# --- Análisis de patrones de Markov ---
def construir_matriz_transicion(secuencia):
    """Construye matriz de transición de orden 1"""
    transiciones = {j: {"piedra": 0, "papel": 0, "tijera": 0} for j in JUGADAS}

    for i in range(len(secuencia) - 1):
        actual = secuencia[i]
        siguiente = secuencia[i + 1]
        transiciones[actual][siguiente] += 1

    # Normalizar
    for jugada in JUGADAS:
        total = sum(transiciones[jugada].values())
        if total > 0:
            for sig in JUGADAS:
                transiciones[jugada][sig] /= total

    return transiciones


# --- Feature engineering ultra-avanzado ---
def generar_features(df, secuencia=7):
    X = []
    y = []

    for i in range(secuencia, len(df)):
        fila = df.iloc[i]

        # Secuencias largas
        seq_amigo = [codificar_jugada(df.iloc[i - j - 1]["jugada_j2"]) for j in range(secuencia)]
        seq_ia = [codificar_jugada(df.iloc[i - j - 1]["jugada_j1"]) for j in range(secuencia)]

        # Resultados previos
        resultados = []
        for j in range(secuencia):
            j1 = df.iloc[i - j - 1]["jugada_j1"]
            j2 = df.iloc[i - j - 1]["jugada_j2"]
            if j1 == j2:
                resultados.append(0)
            elif GANA_A[j1] == j2:
                resultados.append(1)
            else:
                resultados.append(-1)

        # Múltiples ventanas de frecuencia
        prev_filas = df.iloc[:i]
        for ventana in [5, 10, 20, len(prev_filas)]:
            ultimas = prev_filas.iloc[-ventana:]["jugada_j2"] if len(prev_filas) >= ventana else prev_filas["jugada_j2"]
            total = len(ultimas)
            if total > 0:
                freq_p = sum(ultimas == "piedra") / total
                freq_a = sum(ultimas == "papel") / total
                freq_t = sum(ultimas == "tijera") / total
                resultados.extend([freq_p, freq_a, freq_t])
            else:
                resultados.extend([0.33, 0.33, 0.33])

        # Detección de ciclos de longitud 2, 3, 4
        for ciclo_len in [2, 3, 4]:
            if i >= ciclo_len * 2:
                ciclo = [df.iloc[i - j - 1]["jugada_j2"] for j in range(ciclo_len)]
                prev_ciclo = [df.iloc[i - ciclo_len - j - 1]["jugada_j2"] for j in range(ciclo_len)]
                resultados.append(1 if ciclo == prev_ciclo else 0)
            else:
                resultados.append(0)

        # Racha actual
        racha = 1
        for j in range(1, min(7, i)):
            if df.iloc[i - j]["jugada_j2"] == df.iloc[i - j - 1]["jugada_j2"]:
                racha += 1
            else:
                break
        resultados.append(racha)

        # Tendencia reciente (últimas 5 vs anteriores 5)
        if i >= 10:
            ultimas_5 = df.iloc[i - 5:i]["jugada_j2"]
            anteriores_5 = df.iloc[i - 10:i - 5]["jugada_j2"]
            for jugada in JUGADAS:
                diff = sum(ultimas_5 == jugada) - sum(anteriores_5 == jugada)
                resultados.append(diff)
        else:
            resultados.extend([0, 0, 0])

        features = seq_amigo + seq_ia + resultados
        X.append(features)
        y.append(codificar_jugada(fila["jugada_j2"]))

    return np.array(X), np.array(y)


# --- Entrenamiento del modelo ---
def entrenar_modelo():
    if not partidas_csv.exists():
        print(f"[!] No se encontró el archivo de datos: {partidas_csv}")
        return

    df = pd.read_csv(partidas_csv)
    jugadas_validas = ["piedra", "papel", "tijera"]
    df = df[df["jugada_j1"].isin(jugadas_validas) & df["jugada_j2"].isin(jugadas_validas)]

    if len(df) < 15:
        print("[!] No hay suficientes datos para entrenar.")
        return

    X, y = generar_features(df, secuencia=7)

    # Gradient Boosting con hiperparámetros optimizados
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    clf.fit(X, y)

    modelo_pkl.parent.mkdir(exist_ok=True)
    with open(modelo_pkl, "wb") as f:
        pickle.dump(clf, f)

    print(f"[IA] Modelo entrenado con {len(df)} partidas. Score: {clf.score(X, y):.2%}")


# --- Clase JugadorIA ULTRA MEJORADA ---
class JugadorIA:
    def __init__(self):
        self.historial_amigo = deque(maxlen=100)
        self.historial_ia = deque(maxlen=100)
        self.resultados = deque(maxlen=100)
        self.matriz_transicion = None
        self.predicciones_correctas = 0
        self.predicciones_totales = 0

        self.modelo = None
        if modelo_pkl.exists():
            try:
                with open(modelo_pkl, "rb") as f:
                    self.modelo = pickle.load(f)
                print("[IA] Modelo cargado.")
            except:
                print("[IA] Error al cargar modelo.")
        else:
            print("[IA] Sin modelo entrenado.")

    def registrar_ronda(self, jugada_amigo, jugada_ia):
        self.historial_amigo.append(jugada_amigo)
        self.historial_ia.append(jugada_ia)

        if jugada_ia == jugada_amigo:
            self.resultados.append(0)
        elif GANA_A[jugada_ia] == jugada_amigo:
            self.resultados.append(1)
        else:
            self.resultados.append(-1)

        # Actualizar matriz de transición
        if len(self.historial_amigo) > 5:
            self.matriz_transicion = construir_matriz_transicion(list(self.historial_amigo))

    def _predecir_con_markov(self):
        """Predicción usando cadena de Markov"""
        if not self.matriz_transicion or len(self.historial_amigo) < 2:
            return None

        ultima = self.historial_amigo[-1]
        probs = self.matriz_transicion[ultima]
        jugada_predicha = max(probs.items(), key=lambda x: x[1])[0]
        confianza = probs[jugada_predicha]

        if confianza > 0.4:
            return PIERDE_CONTRA[jugada_predicha]
        return None

    def _detectar_ciclo_complejo(self):
        """Detecta ciclos de longitud variable"""
        if len(self.historial_amigo) < 8:
            return None

        # Probar ciclos de longitud 2, 3, 4
        for longitud in [2, 3, 4]:
            if len(self.historial_amigo) >= longitud * 3:
                patron = list(self.historial_amigo)[-longitud:]
                prev1 = list(self.historial_amigo)[-(longitud * 2):-longitud]
                prev2 = list(self.historial_amigo)[-(longitud * 3):-(longitud * 2)]

                if patron == prev1 or patron == prev2:
                    # Ciclo detectado, predecir siguiente
                    return PIERDE_CONTRA[patron[0]]

        return None

    def _estrategia_contra_aleatorio(self):
        """Si el oponente juega muy aleatorio, jugar la menos usada recientemente"""
        if len(self.historial_amigo) < 15:
            return None

        ultimas = list(self.historial_amigo)[-15:]
        counter = Counter(ultimas)

        # Si las frecuencias son muy equilibradas (aleatorio)
        frecuencias = [counter[j] / 15 for j in JUGADAS]
        if max(frecuencias) - min(frecuencias) < 0.15:  # Muy balanceado
            # Jugar contra la menos usada (porque probablemente viene)
            menos_usada = min(counter.items(), key=lambda x: x[1])[0]
            return PIERDE_CONTRA[menos_usada]

        return None

    def _estrategia_explotacion(self):
        """Explota sesgos fuertes del oponente"""
        if len(self.historial_amigo) < 20:
            return None

        ultimas = list(self.historial_amigo)[-20:]
        counter = Counter(ultimas)

        # Si hay un sesgo fuerte (>45%), explotarlo
        for jugada in JUGADAS:
            freq = counter[jugada] / 20
            if freq > 0.45:
                return PIERDE_CONTRA[jugada]

        return None

    def _estrategia_respuesta_a_perdidas(self):
        """Si perdemos muchas seguidas, cambiar radicalmente"""
        if len(self.resultados) < 8:
            return None

        ultimas_8 = list(self.resultados)[-8:]
        derrotas_recientes = ultimas_8.count(-1)

        if derrotas_recientes >= 6:
            # Cambiar a lo opuesto de lo que hemos estado jugando
            ultimas_ia = list(self.historial_ia)[-8:]
            menos_jugada_ia = Counter(ultimas_ia).most_common()[-1][0]
            return menos_jugada_ia

        return None

    def _meta_prediccion(self):
        """Combina múltiples predictores con votación ponderada"""
        votos = Counter()

        # Predicción 1: Markov (peso 2)
        pred_markov = self._predecir_con_markov()
        if pred_markov:
            votos[pred_markov] += 2

        # Predicción 2: Frecuencias recientes (peso 1)
        if len(self.historial_amigo) >= 10:
            ultimas = list(self.historial_amigo)[-10:]
            mas_frecuente = Counter(ultimas).most_common(1)[0][0]
            votos[PIERDE_CONTRA[mas_frecuente]] += 1

        # Predicción 3: Anti-racha (peso 1)
        if len(self.historial_amigo) >= 3:
            ultimas_3 = list(self.historial_amigo)[-3:]
            if ultimas_3[0] == ultimas_3[1]:
                votos[PIERDE_CONTRA[ultimas_3[-1]]] += 1

        # Predicción 4: Ciclo (peso 3)
        pred_ciclo = self._detectar_ciclo_complejo()
        if pred_ciclo:
            votos[pred_ciclo] += 3

        if votos:
            return votos.most_common(1)[0][0]
        return None

    def decidir_jugada(self):
        # Estrategias de emergencia primero
        respuesta_perdidas = self._estrategia_respuesta_a_perdidas()
        if respuesta_perdidas:
            return respuesta_perdidas

        # Explotar sesgos fuertes
        explotacion = self._estrategia_explotacion()
        if explotacion:
            return explotacion

        # Detectar si juega aleatorio
        contra_aleatorio = self._estrategia_contra_aleatorio()
        if contra_aleatorio:
            return contra_aleatorio

        # Meta-predicción (combina varios métodos)
        meta = self._meta_prediccion()
        if meta and len(self.historial_amigo) < 30:
            return meta

        # Usar modelo ML si tenemos suficiente historial
        if len(self.historial_amigo) >= 7 and self.modelo:
            try:
                secuencia = 7
                seq_amigo = [codificar_jugada(list(self.historial_amigo)[-i - 1]) for i in range(secuencia)]
                seq_ia = [codificar_jugada(list(self.historial_ia)[-i - 1]) for i in range(secuencia)]

                resultados_features = list(self.resultados)[-secuencia:]
                while len(resultados_features) < secuencia:
                    resultados_features = [0] + resultados_features

                # Frecuencias múltiples ventanas
                features_extra = []
                for ventana in [5, 10, 20, len(self.historial_amigo)]:
                    ultimas = list(self.historial_amigo)[-ventana:] if len(self.historial_amigo) >= ventana else list(
                        self.historial_amigo)
                    total = len(ultimas)
                    for jugada in JUGADAS:
                        features_extra.append(ultimas.count(jugada) / total if total > 0 else 0.33)

                # Ciclos
                for ciclo_len in [2, 3, 4]:
                    features_extra.append(0)

                # Racha
                racha = 1
                for j in range(1, min(7, len(self.historial_amigo))):
                    if list(self.historial_amigo)[-j] == list(self.historial_amigo)[-j - 1]:
                        racha += 1
                    else:
                        break
                features_extra.append(racha)

                # Tendencias
                features_extra.extend([0, 0, 0])

                X_pred = np.array([seq_amigo + seq_ia + resultados_features + features_extra])
                pred_proba = self.modelo.predict_proba(X_pred)[0]
                pred_idx = np.argmax(pred_proba)
                confianza = pred_proba[pred_idx]

                if confianza > 0.38:
                    jugada_predicha = decodificar_jugada(pred_idx)
                    return PIERDE_CONTRA[jugada_predicha]
            except:
                pass

        # Fallback: meta-predicción o aleatorio
        return meta if meta else np.random.choice(JUGADAS)


if __name__ == "__main__":
    entrenar_modelo()