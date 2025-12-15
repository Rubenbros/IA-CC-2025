"""
RPSAI - Modelo Final Optimizado
================================
Target: 50-55%+ winrate
Strategy: Lower confidence threshold + adaptive multi-window detection
"""

import os
import pickle
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "rps_dataset_clean.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"R": 0, "P": 1, "S": 2}
NUM_A_JUGADA = {0: "R", 1: "P", 2: "S"}
GANA_A = {"R": "S", "P": "R", "S": "P"}
PIERDE_CONTRA = {"R": "P", "P": "S", "S": "R"}


def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"✓ Datos cargados: {len(df)} filas")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df['p1_last_move'] != 'NONE'].copy()
    df = df.dropna()
    print(f"✓ Datos preparados: {len(df)} filas limpias")
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['opp_last_move_num'] = df['p1_last_move'].map(JUGADA_A_NUM)
    df['ai_last_move_num'] = df['p2_last_move'].map(JUGADA_A_NUM)

    outcome_map = {'win': -1, 'lose': 1, 'tie': 0, 'NONE': 0}
    df['opp_last_outcome_num'] = df['p1_last_outcome'].map(outcome_map).fillna(0)

    df['opp_rock_freq'] = df['p1_rock_freq']
    df['opp_paper_freq'] = df['p1_paper_freq']
    df['opp_scissors_freq'] = df['p1_scissors_freq']
    df['opp_win_streak'] = df['p1_win_streak']
    df['opp_loss_streak'] = df['p1_loss_streak']
    df['games_played'] = df['game_number']

    df['opp_most_likely_num'] = df[['opp_rock_freq', 'opp_paper_freq', 'opp_scissors_freq']].idxmax(axis=1).map({
        'opp_rock_freq': 0, 'opp_paper_freq': 1, 'opp_scissors_freq': 2
    })

    freqs = df[['opp_rock_freq', 'opp_paper_freq', 'opp_scissors_freq']]
    df['opp_move_variance'] = freqs.var(axis=1)
    df['opp_move_entropy'] = -((freqs * np.log(freqs + 1e-10)).sum(axis=1))
    df['streak_outcome_interaction'] = df['opp_win_streak'] * df['opp_last_outcome_num']

    sorted_freqs = np.sort(freqs.values, axis=1)
    df['freq_gap'] = sorted_freqs[:, -1] - sorted_freqs[:, -2]
    df['is_predictable'] = (df['freq_gap'] > 0.18).astype(int)

    print(f"✓ Features creadas: {len(df.columns)} columnas")
    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        'opp_last_move_num', 'ai_last_move_num',
        'opp_rock_freq', 'opp_paper_freq', 'opp_scissors_freq',
        'opp_win_streak', 'opp_loss_streak', 'opp_last_outcome_num',
        'games_played', 'opp_most_likely_num',
        'opp_move_variance', 'opp_move_entropy',
        'streak_outcome_interaction', 'freq_gap', 'is_predictable'
    ]

    X = df[feature_cols].fillna(0)
    y = df['p1_current_move'].map(JUGADA_A_NUM)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"✓ Features: {len(feature_cols)} | Samples: {len(X)}")
    return X, y


def entrenar_modelo(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    modelos = {
        'RandomForest': RandomForestClassifier(
            n_estimators=600,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=400,
            max_depth=15,
            learning_rate=0.18,
            subsample=0.85,
            random_state=42
        )
    }

    mejor_modelo = None
    mejor_acc = 0.0

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{nombre}: Test Accuracy = {acc:.3f}")

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n✓ Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)


class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.historial = []

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✓ Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠️ Modelo no encontrado. Entrena primero con: python src/modelo.py")

    def registrar_ronda(self, jugada_oponente: str, jugada_ia: str):
        self.historial.append((jugada_oponente, jugada_ia))

    def detectar_sesgo_adaptativo(self) -> str:
        """Detecta sesgos con múltiples ventanas y umbrales adaptativos."""
        if len(self.historial) < 6:
            return None

        # Probar múltiples ventanas
        ventanas = [8, 10, 12]
        mejor_prediccion = None
        mejor_confianza = 0

        for ventana in ventanas:
            if len(self.historial) < ventana:
                continue

            recent = [j_opp for j_opp, _ in self.historial[-ventana:]]
            counter = Counter(recent)
            total = len(recent)

            for move, count in counter.most_common(1):
                freq = count / total

                # Umbral adaptativo basado en cuántos datos tenemos
                if len(self.historial) >= 15:
                    threshold = 0.40  # Más datos = más conservador
                elif len(self.historial) >= 10:
                    threshold = 0.38
                else:
                    threshold = 0.42  # Pocos datos = necesitamos más evidencia

                if freq > threshold and freq > mejor_confianza:
                    mejor_confianza = freq
                    mejor_prediccion = move

        return mejor_prediccion

    def obtener_features_actuales(self) -> np.ndarray:
        if len(self.historial) == 0:
            return np.zeros(15)

        opp_moves = [j_opp for j_opp, _ in self.historial]
        ai_moves = [j_ia for _, j_ia in self.historial]

        opp_last = JUGADA_A_NUM.get(opp_moves[-1], 0)
        ai_last = JUGADA_A_NUM.get(ai_moves[-1], 0)

        total = len(opp_moves)
        opp_rock_freq = opp_moves.count('R') / total
        opp_paper_freq = opp_moves.count('P') / total
        opp_scissors_freq = opp_moves.count('S') / total

        opp_win_streak = 0
        opp_loss_streak = 0
        opp_last_outcome = 0

        if len(self.historial) > 0:
            last_opp, last_ai = self.historial[-1]
            if GANA_A[last_opp] == last_ai:
                opp_last_outcome = 1
                for j_opp, j_ai in reversed(self.historial):
                    if GANA_A[j_opp] == j_ai:
                        opp_win_streak += 1
                    else:
                        break
            elif GANA_A[last_ai] == last_opp:
                opp_last_outcome = -1
                for j_opp, j_ai in reversed(self.historial):
                    if GANA_A[j_ai] == j_opp:
                        opp_loss_streak += 1
                    else:
                        break

        games_played = len(self.historial)
        opp_freqs = [opp_rock_freq, opp_paper_freq, opp_scissors_freq]
        opp_most_likely = opp_freqs.index(max(opp_freqs))

        opp_move_variance = np.var(opp_freqs)
        opp_move_entropy = -sum(p * np.log(p + 1e-10) for p in opp_freqs)
        streak_outcome_interaction = opp_win_streak * opp_last_outcome
        sorted_freqs = sorted(opp_freqs, reverse=True)
        freq_gap = sorted_freqs[0] - sorted_freqs[1]
        is_predictable = 1 if freq_gap > 0.18 else 0

        return np.array([
            opp_last, ai_last,
            opp_rock_freq, opp_paper_freq, opp_scissors_freq,
            opp_win_streak, opp_loss_streak, opp_last_outcome,
            games_played, opp_most_likely,
            opp_move_variance, opp_move_entropy,
            streak_outcome_interaction, freq_gap, is_predictable
        ])

    def predecir_con_modelo(self) -> str:
        if self.modelo is None or len(self.historial) < 3:
            return None

        features = self.obtener_features_actuales()

        # Usar probabilidades
        if hasattr(self.modelo, 'predict_proba'):
            probs = self.modelo.predict_proba([features])[0]
            max_prob = np.max(probs)

            # CLAVE: Umbral más bajo (36% en vez de 40%)
            if max_prob > 0.36:
                prediccion = np.argmax(probs)
                return NUM_A_JUGADA[prediccion]

        return None

    def predecir_jugada_oponente(self) -> str:
        """Predicción balanceada con umbrales optimizados."""

        # Early game: conservative
        if len(self.historial) < 3:
            return np.random.choice(["R", "P", "S"])

        # PRIORITY 1: Adaptive bias detection (múltiples ventanas)
        if len(self.historial) >= 8:
            sesgo = self.detectar_sesgo_adaptativo()
            if sesgo:
                return sesgo

        # PRIORITY 2: ML model (con umbral más bajo)
        ml_pred = self.predecir_con_modelo()
        if ml_pred:
            return ml_pred

        # PRIORITY 3: Simple overall frequency (con umbral bajo)
        if len(self.historial) >= 6:
            all_moves = [j_opp for j_opp, _ in self.historial]
            counter = Counter(all_moves)
            most_common, count = counter.most_common(1)[0]
            freq = count / len(all_moves)

            # Umbral 38%
            if freq > 0.38:
                return most_common

        # Fallback: random
        return np.random.choice(["R", "P", "S"])

    def decidir_jugada(self) -> str:
        prediccion_oponente = self.predecir_jugada_oponente()

        # 8% randomización
        if np.random.random() < 0.08:
            return np.random.choice(["R", "P", "S"])

        return PIERDE_CONTRA[prediccion_oponente]


def main():
    print("=" * 60)
    print(" RPSAI - Entrenamiento Final Optimizado")
    print("=" * 60)

    print("\n[1/5] Cargando datos...")
    df = cargar_datos()

    print("\n[2/5] Preparando datos...")
    df = preparar_datos(df)

    print("\n[3/5] Creando features...")
    df = crear_features(df)

    print("\n[4/5] Entrenando modelo...")
    X, y = seleccionar_features(df)
    modelo = entrenar_modelo(X, y)

    print("\n[5/5] Guardando modelo...")
    guardar_modelo(modelo)

    print("\n" + "=" * 60)
    print(" ✓ Entrenamiento completado!")
    print("=" * 60)
    print("\nAhora ejecuta: python src/evaluador.py")


if __name__ == "__main__":
    main()
