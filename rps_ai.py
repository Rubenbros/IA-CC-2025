"""
IA de Piedra, Papel o Tijera
Aprende de los datos recopilados por los usuarios para predecir jugadas.
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Mapeo de nombres a valores estándar
MOVE_MAP = {
    'piedra': 'piedra', 'Piedra': 'piedra', 'r': 'piedra', 'R': 'piedra', 'rock': 'piedra', 'Rock': 'piedra',
    '1': 'piedra',
    'papel': 'papel', 'Papel': 'papel', 'p': 'papel', 'P': 'papel', 'paper': 'papel', 'Paper': 'papel',
    '2': 'papel',
    'tijera': 'tijera', 'Tijera': 'tijera', 'tijeras': 'tijera', 'Tijeras': 'tijera',
    's': 'tijera', 'S': 'tijera', 'scissors': 'tijera', 'Scissors': 'tijera',
    '3': 'tijera'
}

# Lo que gana a cada cosa
BEATS = {
    'piedra': 'papel',
    'papel': 'tijera',
    'tijera': 'piedra'
}

class RPSDataLoader:
    """Carga y unifica datos de todos los CSVs disponibles."""

    def __init__(self, base_path):
        self.base_path = base_path
        self.all_moves = []
        self.sequences = []  # Secuencias de jugadas para análisis

    def normalize_move(self, move):
        """Normaliza el nombre de una jugada."""
        if pd.isna(move):
            return None
        move_str = str(move).strip().lower()
        for key, val in MOVE_MAP.items():
            if move_str == key.lower():
                return val
        return None

    def load_all_data(self):
        """Carga todos los CSVs encontrados."""
        patterns = [
            os.path.join(self.base_path, 'rps-ai-submissions', '**', 'data', '*.csv'),
            os.path.join(self.base_path, 'rps-ai-submissions', '**', '*.csv'),
        ]

        csv_files = set()
        for pattern in patterns:
            csv_files.update(glob.glob(pattern, recursive=True))

        print(f"Encontrados {len(csv_files)} archivos CSV")

        total_rows = 0
        for csv_file in csv_files:
            rows = self._load_csv(csv_file)
            total_rows += rows

        print(f"Total de jugadas cargadas: {total_rows}")
        return self.all_moves

    def _load_csv(self, filepath):
        """Intenta cargar un CSV y extraer las jugadas."""
        try:
            # Intentar diferentes encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except:
                    continue
            else:
                return 0

            # Buscar columnas de jugadas
            j1_cols = ['jugada_j1', 'jugador1', 'jugador', 'Jugador 1', 'p1_current_move',
                       'movimiento_j1', 'jugada_jugador1', 'player1', 'move1', 'j1']
            j2_cols = ['jugada_j2', 'jugador2', 'IA', 'ia', 'Jugador 2', 'p2_last_move',
                       'movimiento_j2', 'jugada_jugador2', 'player2', 'move2', 'j2']

            j1_col = None
            j2_col = None

            for col in j1_cols:
                if col in df.columns:
                    j1_col = col
                    break

            for col in j2_cols:
                if col in df.columns:
                    j2_col = col
                    break

            if j1_col is None:
                return 0

            # Extraer secuencia de jugadas del jugador 1 (humano)
            sequence = []
            rows_loaded = 0

            for _, row in df.iterrows():
                move1 = self.normalize_move(row.get(j1_col))
                move2 = self.normalize_move(row.get(j2_col)) if j2_col else None

                if move1:
                    self.all_moves.append({
                        'j1': move1,
                        'j2': move2,
                        'source': filepath
                    })
                    sequence.append(move1)
                    rows_loaded += 1

            if len(sequence) >= 3:
                self.sequences.append(sequence)

            return rows_loaded

        except Exception as e:
            return 0


class MLPredictor:
    """Predictor basado en Machine Learning (Random Forest) con features avanzadas."""

    def __init__(self, lookback=5):
        self.lookback = lookback
        self.model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['piedra', 'papel', 'tijera'])
        self.is_trained = False
        self.feature_names = []  # Para debugging

    def _one_hot_move(self, move):
        """Convierte jugada a one-hot."""
        return [1 if move == 'piedra' else 0,
                1 if move == 'papel' else 0,
                1 if move == 'tijera' else 0]

    def _extract_features(self, sequence, idx, ia_sequence=None, results=None):
        """
        Extrae features avanzadas para predicción.

        Features (total ~45):
        - Últimas 5 jugadas humano (15)
        - Últimas 3 jugadas IA si disponible (9)
        - Últimos 3 resultados si disponible (9)
        - Frecuencias acumuladas (3)
        - Racha actual (4)
        - Features WSLS (3)
        - Detección de cycling (2)
        - Entropía reciente (1)
        """
        features = []

        # === 1. Últimas N jugadas del humano (one-hot) ===
        for i in range(self.lookback):
            pos = idx - i - 1
            if pos >= 0:
                features.extend(self._one_hot_move(sequence[pos]))
            else:
                features.extend([0, 0, 0])

        # === 2. Últimas 3 jugadas de la IA (one-hot) ===
        if ia_sequence:
            for i in range(3):
                pos = idx - i - 1
                if pos >= 0 and pos < len(ia_sequence):
                    features.extend(self._one_hot_move(ia_sequence[pos]))
                else:
                    features.extend([0, 0, 0])
        else:
            features.extend([0] * 9)

        # === 3. Últimos 3 resultados (one-hot: win/lose/tie) ===
        if results:
            for i in range(3):
                pos = idx - i - 1
                if pos >= 0 and pos < len(results):
                    r = results[pos]
                    features.extend([1 if r == 'win' else 0,
                                   1 if r == 'lose' else 0,
                                   1 if r == 'tie' else 0])
                else:
                    features.extend([0, 0, 0])
        else:
            features.extend([0] * 9)

        # === 4. Frecuencias acumuladas ===
        history = sequence[:idx]
        if len(history) > 0:
            counts = Counter(history)
            total = len(history)
            features.append(counts.get('piedra', 0) / total)
            features.append(counts.get('papel', 0) / total)
            features.append(counts.get('tijera', 0) / total)
        else:
            features.extend([0.33, 0.33, 0.33])

        # === 5. Racha actual ===
        if idx > 0:
            last = sequence[idx-1]
            streak = 1
            for i in range(idx - 2, -1, -1):
                if sequence[i] == last:
                    streak += 1
                else:
                    break
            features.append(min(streak / 5, 1.0))  # Normalizado, cap en 5
            features.extend(self._one_hot_move(last))
        else:
            features.extend([0, 0, 0, 0])

        # === 6. Features WSLS (Win-Stay, Lose-Shift) ===
        if results and len(results) >= 3:
            # Tasa de "stay" después de ganar
            wins_before = [(i, sequence[i]) for i in range(min(idx-1, len(results)-1))
                          if results[i] == 'win']
            if len(wins_before) >= 2:
                stay_after_win = sum(1 for i, m in wins_before
                                    if i+1 < len(sequence) and sequence[i+1] == m) / len(wins_before)
            else:
                stay_after_win = 0.5

            # Tasa de "shift" después de perder
            losses_before = [(i, sequence[i]) for i in range(min(idx-1, len(results)-1))
                            if results[i] == 'lose']
            if len(losses_before) >= 2:
                shift_after_lose = sum(1 for i, m in losses_before
                                      if i+1 < len(sequence) and sequence[i+1] != m) / len(losses_before)
            else:
                shift_after_lose = 0.5

            # Último resultado
            last_result_idx = min(idx-1, len(results)-1)
            if last_result_idx >= 0:
                last_res = results[last_result_idx]
                features.append(1 if last_res == 'win' else 0)
                features.append(1 if last_res == 'lose' else 0)
            else:
                features.extend([0, 0])

            features.append(stay_after_win)
            features.append(shift_after_lose)
        else:
            features.extend([0, 0, 0.5, 0.5])

        # === 7. Detección de cycling ===
        move_to_num = {'piedra': 0, 'papel': 1, 'tijera': 2}
        if idx >= 3:
            recent = [move_to_num[sequence[i]] for i in range(idx-3, idx)]
            # Cycling up: piedra(0)→papel(1)→tijera(2) = +1 mod 3
            ups = sum(1 for i in range(len(recent)-1) if (recent[i+1] - recent[i]) % 3 == 1)
            # Cycling down: piedra(0)→tijera(2)→papel(1) = -1 mod 3
            downs = sum(1 for i in range(len(recent)-1) if (recent[i+1] - recent[i]) % 3 == 2)
            features.append(ups / 2)  # Normalizado
            features.append(downs / 2)
        else:
            features.extend([0, 0])

        # === 8. Entropía reciente (predictibilidad) ===
        recent = sequence[max(0, idx-10):idx]
        if len(recent) >= 3:
            counts = Counter(recent)
            probs = [c/len(recent) for c in counts.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            features.append(entropy / np.log2(3))  # Normalizado [0,1]
        else:
            features.append(1.0)  # Máxima entropía si pocos datos

        return features

    def train(self, sequences):
        """Entrena el modelo con secuencias de jugadas."""
        X = []
        y = []

        for sequence in sequences:
            if len(sequence) < self.lookback + 1:
                continue

            for i in range(self.lookback, len(sequence)):
                # Sin IA ni resultados para datos históricos
                features = self._extract_features(sequence, i)
                target = sequence[i]
                X.append(features)
                y.append(target)

        if len(X) < 100:
            print(f"  ML: Datos insuficientes ({len(X)} muestras)")
            return

        X = np.array(X)
        y = self.label_encoder.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        print(f"  ML: Entrenado con {len(X_train)} muestras ({len(X[0])} features)")
        print(f"  ML: Accuracy train={train_acc:.1%}, test={test_acc:.1%}")

    def predict_probabilities(self, history, ia_history=None, results=None):
        """Predice probabilidades usando el modelo ML."""
        if not self.is_trained or len(history) < self.lookback:
            return {'piedra': 1/3, 'papel': 1/3, 'tijera': 1/3}

        sequence = history + ['piedra']  # dummy
        features = self._extract_features(sequence, len(history), ia_history, results)
        features = np.array(features).reshape(1, -1)

        proba = self.model.predict_proba(features)[0]
        classes = self.label_encoder.classes_

        return {classes[i]: proba[i] for i in range(len(classes))}


class MarkovPredictor:
    """Predictor basado en cadenas de Markov de diferentes órdenes."""

    def __init__(self, max_order=3):
        self.max_order = max_order
        self.transitions = {}  # {orden: {contexto: {siguiente: count}}}

        for order in range(1, max_order + 1):
            self.transitions[order] = defaultdict(lambda: defaultdict(int))

    def train(self, sequences):
        """Entrena el modelo con secuencias de jugadas."""
        for sequence in sequences:
            for order in range(1, self.max_order + 1):
                for i in range(len(sequence) - order):
                    context = tuple(sequence[i:i+order])
                    next_move = sequence[i+order]
                    self.transitions[order][context][next_move] += 1

        print(f"Modelo Markov entrenado:")
        for order in range(1, self.max_order + 1):
            print(f"  Orden {order}: {len(self.transitions[order])} contextos")

    def predict_probabilities(self, history):
        """Predice probabilidades del siguiente movimiento dado el historial."""
        probs = {'piedra': 0, 'papel': 0, 'tijera': 0}
        total_weight = 0

        for order in range(self.max_order, 0, -1):
            if len(history) >= order:
                context = tuple(history[-order:])
                if context in self.transitions[order]:
                    counts = self.transitions[order][context]
                    total = sum(counts.values())
                    weight = order * 2  # Mayor peso a órdenes más altos

                    for move, count in counts.items():
                        probs[move] += (count / total) * weight
                    total_weight += weight

        if total_weight > 0:
            for move in probs:
                probs[move] /= total_weight
        else:
            # Sin datos, usar distribución uniforme
            probs = {'piedra': 1/3, 'papel': 1/3, 'tijera': 1/3}

        return probs


class PatternAnalyzer:
    """Analiza patrones comunes en las jugadas."""

    def __init__(self):
        self.win_responses = defaultdict(lambda: defaultdict(int))  # Qué juega después de ganar con X
        self.lose_responses = defaultdict(lambda: defaultdict(int))  # Qué juega después de perder con X
        self.tie_responses = defaultdict(lambda: defaultdict(int))   # Qué juega después de empatar con X
        self.overall_freq = Counter()

    def analyze(self, moves_data, sequences):
        """Analiza los patrones de los datos."""
        # Frecuencia general
        for move in moves_data:
            if move['j1']:
                self.overall_freq[move['j1']] += 1

        total = sum(self.overall_freq.values())
        print("\nFrecuencia general de jugadas humanas:")
        for move, count in self.overall_freq.most_common():
            print(f"  {move}: {count} ({100*count/total:.1f}%)")

        # Analizar respuestas después de resultados
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_move = sequence[i+1]
                # Simplificación: asumimos que el humano tiende a repetir patrones
                self.tie_responses[current][next_move] += 1

        return self.overall_freq


class RPSAIPlayer:
    """IA que juega piedra, papel o tijera."""

    def __init__(self):
        self.markov = MarkovPredictor(max_order=4)
        self.ml_predictor = MLPredictor(lookback=5)
        self.pattern_analyzer = PatternAnalyzer()
        self.history = []  # Historial de jugadas del oponente
        self.my_history = []  # Mis propias jugadas
        self.results = []  # Historial de resultados (win/lose/tie desde perspectiva humano)
        self.overall_freq = Counter()
        self.wins = 0
        self.losses = 0
        self.ties = 0

    def train(self, base_path):
        """Entrena la IA con los datos disponibles."""
        print("=" * 50)
        print("ENTRENANDO IA DE PIEDRA, PAPEL O TIJERA")
        print("=" * 50)

        loader = RPSDataLoader(base_path)
        moves_data = loader.load_all_data()

        # Entrenar Markov
        print("\nEntrenando modelo Markov...")
        self.markov.train(loader.sequences)

        # Entrenar ML (Random Forest)
        print("\nEntrenando modelo ML (Random Forest)...")
        self.ml_predictor.train(loader.sequences)

        # Analizar patrones
        self.overall_freq = self.pattern_analyzer.analyze(moves_data, loader.sequences)

        print("\nIA entrenada y lista para jugar!")
        print("=" * 50)

    def predict_opponent_move(self):
        """Predice el siguiente movimiento del oponente con features avanzadas."""
        n = len(self.history)

        if n == 0:
            # Primera jugada: usar frecuencia general (sesgo hacia piedra conocido)
            total = sum(self.overall_freq.values())
            if total > 0:
                probs = {m: c/total for m, c in self.overall_freq.items()}
            else:
                probs = {'piedra': 0.36, 'papel': 0.32, 'tijera': 0.32}  # Sesgo piedra
        elif n < 5:
            # Pocas jugadas: mezclar ML con frecuencia general
            ml_probs = self.ml_predictor.predict_probabilities(
                self.history, self.my_history, self.results
            )
            total_freq = sum(self.overall_freq.values())
            gen_probs = {m: self.overall_freq.get(m, 1)/total_freq for m in ['piedra', 'papel', 'tijera']}

            weight = n / 5
            probs = {m: ml_probs[m] * weight + gen_probs[m] * (1-weight) for m in ['piedra', 'papel', 'tijera']}
        else:
            # === Predicción avanzada con múltiples señales ===

            # 1. ML con features completas (40% peso)
            ml_probs = self.ml_predictor.predict_probabilities(
                self.history, self.my_history, self.results
            )

            # 2. Frecuencia del jugador actual (25% peso)
            counts = Counter(self.history)
            player_probs = {m: counts.get(m, 0) / n for m in ['piedra', 'papel', 'tijera']}

            # 3. Transiciones después de última jugada (20% peso)
            last = self.history[-1]
            transitions = [self.history[i+1] for i in range(n-1) if self.history[i] == last]
            if len(transitions) >= 2:
                tc = Counter(transitions)
                trans_probs = {m: tc.get(m, 0) / len(transitions) for m in ['piedra', 'papel', 'tijera']}
            else:
                trans_probs = player_probs

            # 4. WSLS: Predicción basada en último resultado (15% peso)
            wsls_probs = {'piedra': 1/3, 'papel': 1/3, 'tijera': 1/3}
            if self.results:
                last_result = self.results[-1]
                last_human = self.history[-1]

                if last_result == 'win':
                    # Humano ganó → probablemente repita (win-stay)
                    wsls_probs[last_human] = 0.5
                    for m in ['piedra', 'papel', 'tijera']:
                        if m != last_human:
                            wsls_probs[m] = 0.25
                elif last_result == 'lose':
                    # Humano perdió → probablemente cambie (lose-shift)
                    # Muchos cambian a lo que les habría ganado
                    would_have_won = BEATS[self.my_history[-1]]  # Lo que gana a lo que jugué
                    wsls_probs[last_human] = 0.2
                    wsls_probs[would_have_won] = 0.5
                    remaining = [m for m in ['piedra', 'papel', 'tijera'] if m not in [last_human, would_have_won]]
                    if remaining:
                        wsls_probs[remaining[0]] = 0.3

            # Combinar todas las señales
            probs = {}
            for m in ['piedra', 'papel', 'tijera']:
                probs[m] = (
                    ml_probs[m] * 0.40 +
                    player_probs[m] * 0.25 +
                    trans_probs[m] * 0.20 +
                    wsls_probs[m] * 0.15
                )

        # Ruido mínimo para evitar ser totalmente predecible
        noise = 0.03
        probs = {m: probs[m] * (1-noise) + noise/3 for m in ['piedra', 'papel', 'tijera']}

        # Normalizar
        total = sum(probs.values())
        probs = {m: p/total for m, p in probs.items()}

        return probs

    def choose_move(self):
        """Elige el mejor movimiento para ganar."""
        predicted_probs = self.predict_opponent_move()

        # Calcular valor esperado de cada movimiento
        expected_values = {}
        for my_move in ['piedra', 'papel', 'tijera']:
            ev = 0
            for opp_move, prob in predicted_probs.items():
                if BEATS[opp_move] == my_move:
                    ev += prob * 1  # Victoria
                elif BEATS[my_move] == opp_move:
                    ev += prob * (-1)  # Derrota
                # Empate = 0
            expected_values[my_move] = ev

        # Elegir movimiento con mejor valor esperado
        best_move = max(expected_values, key=expected_values.get)

        # Solo añadir aleatoriedad si la ventaja es muy pequeña (decisión incierta)
        best_ev = expected_values[best_move]
        moves = ['piedra', 'papel', 'tijera']
        moves.remove(best_move)
        second_best = max(moves, key=lambda m: expected_values[m])
        second_ev = expected_values[second_best]

        # Si la diferencia es < 0.05, hay incertidumbre, añadir algo de variación
        if abs(best_ev - second_ev) < 0.05 and random.random() < 0.3:
            best_move = second_best

        return best_move, predicted_probs, expected_values

    def register_opponent_move(self, move):
        """Registra el movimiento del oponente."""
        normalized = MOVE_MAP.get(move.lower(), move.lower())
        self.history.append(normalized)
        return normalized

    def register_my_move(self, move):
        """Registra mi propio movimiento."""
        self.my_history.append(move)

    def get_result(self, my_move, opp_move):
        """Determina el resultado de la partida y registra para features."""
        if my_move == opp_move:
            self.ties += 1
            self.results.append('tie')  # Desde perspectiva del humano
            return "empate"
        elif BEATS[opp_move] == my_move:
            self.wins += 1
            self.results.append('lose')  # Humano perdió
            return "gano_ia"
        else:
            self.losses += 1
            self.results.append('win')  # Humano ganó
            return "gana_humano"

    def get_stats(self):
        """Devuelve estadísticas de la partida (sin contar empates en winrate)."""
        total_decisivas = self.wins + self.losses
        if total_decisivas == 0:
            return f"IA: {self.wins}W-{self.losses}L ({self.ties} empates) - Sin partidas decisivas"
        winrate = 100 * self.wins / total_decisivas
        return f"IA: {self.wins}W-{self.losses}L ({self.ties} empates) | Winrate: {winrate:.1f}%"

    def save_model(self, filepath):
        """Guarda el modelo entrenado."""
        # Convertir defaultdicts a dicts normales para pickle
        markov_data = {}
        for order, transitions in self.markov.transitions.items():
            markov_data[order] = {k: dict(v) for k, v in transitions.items()}

        data = {
            'markov_transitions': markov_data,
            'overall_freq': dict(self.overall_freq)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Modelo guardado en {filepath}")

    def load_model(self, filepath):
        """Carga un modelo entrenado."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        # Restaurar transiciones de Markov
        for order, transitions in data['markov_transitions'].items():
            self.markov.transitions[order] = defaultdict(lambda: defaultdict(int))
            for context, counts in transitions.items():
                self.markov.transitions[order][context] = defaultdict(int, counts)
        self.overall_freq = Counter(data['overall_freq'])
        print(f"Modelo cargado desde {filepath}")


def clear_screen():
    """Limpia la pantalla."""
    os.system('cls' if os.name == 'nt' else 'clear')


# ASCII Art para cada jugada
ASCII_PIEDRA = r"""
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
"""

ASCII_PAPEL = r"""
     _______
---'    ____)____
           ______)
          _______)
         _______)
---.__________)
"""

ASCII_TIJERA = r"""
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
"""

def mostrar_opciones():
    """Muestra las opciones con ASCII art."""
    print("""
+-------------+-------------+-------------+
|   [1]       |   [2]       |   [3]       |
|  PIEDRA     |   PAPEL     |  TIJERA     |
+-------------+-------------+-------------+
|    _____    |    _____    |    _____    |
|---'   __)   |---'    __)_ |---'   __)__ |
|     (____)  |        ____)|       _____)|
|     (____)  |       _____)| ___________)|
|     (___)   |      _____))|     (____)  |
|---._(___)   |---.______)) |---._(___)   |
+-------------+-------------+-------------+
    """)


def mostrar_resultado_visual(jugador, ia):
    """Muestra el resultado con ASCII art lado a lado."""
    arts = {
        'piedra': [
            "    _____    ",
            "---'   __)   ",
            "     (____)  ",
            "     (____)  ",
            "     (___)   ",
            "---._(___)   "
        ],
        'papel': [
            "    _____    ",
            "---'    __)_ ",
            "        ____)",
            "       _____)",
            "      _____) ",
            "---.______)) "
        ],
        'tijera': [
            "    _____    ",
            "---'   __)__ ",
            "       _____)",
            " ___________)",
            "     (____)  ",
            "---._(___)   "
        ]
    }

    j_art = arts.get(jugador, arts['piedra'])
    ia_art = arts.get(ia, arts['piedra'])

    print(f"\n      TU ({jugador.upper()})         vs        IA ({ia.upper()})")
    print("  " + "="*50)
    for i in range(len(j_art)):
        print(f"  {j_art[i]}      {ia_art[i]}")
    print("  " + "="*50)


def play_interactive(ai, time_limit=3):
    """Modo de juego interactivo con timer."""
    import threading

    print("\n" + "=" * 50)
    print("       PIEDRA, PAPEL O TIJERA vs IA")
    print("=" * 50)
    if time_limit < 100:
        print(f"      Tienes {time_limit} segundos para elegir!")
    print("            [0] = Salir")
    print("=" * 50)

    mostrar_opciones()

    input("  Presiona ENTER para comenzar...")

    round_num = 0

    while True:
        round_num += 1

        # IA elige ANTES
        ai_move, predicted_probs, expected_values = ai.choose_move()

        # Mostrar ronda
        print(f"\n{'='*50}")
        print(f"             RONDA {round_num}")
        print(f"{'='*50}")
        mostrar_opciones()

        # Input con timer
        user_input = None

        def get_input():
            nonlocal user_input
            try:
                user_input = input("  >>> Tu jugada [1/2/3]: ").strip().lower()
            except:
                pass

        if time_limit < 100:
            # Con timer
            print(f"  Tiempo: ", end='', flush=True)
            input_thread = threading.Thread(target=get_input, daemon=True)
            input_thread.start()

            for remaining in range(time_limit, 0, -1):
                if not input_thread.is_alive():
                    break
                print(f" {remaining} ", end='', flush=True)
                input_thread.join(timeout=1)

            if input_thread.is_alive():
                user_input = random.choice(['1', '2', '3'])
                print(f"\n\n  TIEMPO AGOTADO! Jugada aleatoria: {MOVE_MAP[user_input].upper()}")
            else:
                print()
        else:
            # Sin timer
            get_input()

        # Validar salida
        if user_input in ['salir', 'exit', 'q', '0']:
            print(f"\n{'='*50}")
            print("           RESULTADO FINAL")
            print(f"{'='*50}")
            print(f"  {ai.get_stats()}")
            total = ai.wins + ai.losses
            if total > 0:
                print()
                if ai.wins > ai.losses:
                    print("      >>> LA IA GANA EL MATCH <<<")
                elif ai.losses > ai.wins:
                    print("      >>>   TU GANAS EL MATCH  <<<")
                else:
                    print("      >>> EMPATE EN EL MATCH <<<")
            print(f"{'='*50}\n")
            break

        if user_input not in MOVE_MAP:
            print("  Jugada no valida! Usa 1, 2 o 3")
            round_num -= 1
            continue

        # Registrar jugadas
        user_move = ai.register_opponent_move(user_input)
        ai.register_my_move(ai_move)

        # Resultado
        result = ai.get_result(ai_move, user_move)

        # Mostrar resultado visual
        mostrar_resultado_visual(user_move, ai_move)

        if result == "empate":
            print("\n            ** EMPATE **")
        elif result == "gano_ia":
            print("\n          ** GANA LA IA **")
        else:
            print("\n          ** GANASTE!! **")

        print(f"\n  {ai.get_stats()}")


def main():
    """Función principal."""
    base_path = os.path.dirname(os.path.abspath(__file__))

    print("""
    ____  ____  ____     ___    ____
   / __ \\/ __ \\/ __ \\   /   |  /  _/
  / /_/ / /_/ / / / /  / /| |  / /
 / _, _/ ____/ /_/ /  / ___ |_/ /
/_/ |_/_/    \\____/  /_/  |_/___/

  PIEDRA - PAPEL - TIJERA vs IA
    """)

    ai = RPSAIPlayer()
    ai.train(base_path)

    # Guardar modelo entrenado
    model_path = os.path.join(base_path, 'rps_model.pkl')
    ai.save_model(model_path)

    # Elegir modo
    print("\n" + "="*50)
    print("  ELIGE MODO DE JUEGO")
    print("="*50)
    print("  1. Modo RAPIDO (2 segundos)")
    print("  2. Modo NORMAL (3 segundos)")
    print("  3. Modo TRANQUILO (5 segundos)")
    print("  4. Modo SIN TIMER (sin limite)")
    print("="*50)

    mode = input("\n  Elige modo (1-4): ").strip()

    time_limits = {'1': 2, '2': 3, '3': 5, '4': 999}
    time_limit = time_limits.get(mode, 3)

    if mode == '4':
        print("\n  Modo sin timer activado.")
        time_limit = 999  # Practicamente sin limite

    # Jugar
    play_interactive(ai, time_limit=time_limit)


if __name__ == "__main__":
    main()
