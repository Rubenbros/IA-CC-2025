import os
import random
import keyboard
import time
import pandas as pd
from datetime import datetime

# Controles
PLAYER1_KEYS = {'q': 'piedra', 'w': 'papel', 'e': 'tijera'}
PLAYER2_KEYS = {'i': 'piedra', 'o': 'papel', 'p': 'tijera'}
# Donde guardar las partidas
FILE_NAME = 'data/partidas.csv'


def get_result(p1: str, p2: str) -> str:
    """
    Determina el resultado de la partida desde la perspectiva del jugador1
    """
    if p1 == p2:
        return 'empate'
    if (p1 == 'piedra' and p2 == 'tijera') or (p1 == 'papel' and p2 == 'piedra') or (p1 == 'tijera' and p2 == 'papel'):
        return 'victoria'
    return 'derrota'


def get_moves_within(seconds: int = 3):
    """
    Guarda las jugadas de cada jugador durante un tiempo limitado, y tiempo de reacción.
    """
    p1_move = None
    p2_move = None
    p1_time_ms = None
    p2_time_ms = None

    start = time.time()
    while time.time() - start < seconds:
        event = keyboard.read_event(suppress=False)

        if event.event_type != keyboard.KEY_DOWN:
            continue

        key = event.name.lower()
        now = time.time()
        elapsed_ms = int((now - start) * 1000)

        if key in PLAYER1_KEYS and p1_move is None:
            p1_move = PLAYER1_KEYS[key]
            p1_time_ms = elapsed_ms
        elif key in PLAYER2_KEYS and p2_move is None:
            p2_move = PLAYER2_KEYS[key]
            p2_time_ms = elapsed_ms

        if p1_move is not None and p2_move is not None:
            break

    return p1_move, p2_move, p1_time_ms, p2_time_ms


def load_existing_data():
    """
    Carga el CSV con los datos (si existe)
    """
    columns = [
        'session_id',
        'numero_ronda',
        'jugada_j1',
        'jugada_j2',
        'resultado',
        'timestamp',
        'tiempo_j1_ms',
        'tiempo_j2_ms'
    ]

    if os.path.exists(FILE_NAME):
        df = pd.read_csv(FILE_NAME)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        if not df.empty and 'numero_ronda' in df.columns:
            next_id = int(df['numero_ronda'].max()) + 1
        else:
            next_id = 1
    else:
        df = pd.DataFrame(columns=columns)
        next_id = 1

    return df, next_id


def main():
    session_id = random.randint(1_000_000, 9_999_999)

    df, game_id = load_existing_data()

    print(f"ID Sesión: {session_id}")
    print(f"Jugando ronda #: {game_id}")

    try:
        while True:
            print("\nLa próxima ronda empieza en:")
            for i in range(3, 0, -1):
                print(i)
                time.sleep(1)

            print("A jugar! (P1: q/w/e, P2: i/o/p)")
            p1_move, p2_move, p1_time_ms, p2_time_ms = get_moves_within(3)

            if p1_move is None or p2_move is None:
                print("Uno o varios jugadores no jugaron a tiempo, omitiendo ronda.")
                continue

            result = get_result(p1_move, p2_move)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            print(f"Round {game_id}")
            print(
                f"J1: {p1_move} ({p1_time_ms}ms) | "
                f"J2: {p2_move} ({p2_time_ms}ms) -> {result.upper()}"
            )

            df.loc[len(df)] = [
                session_id,
                game_id,
                p1_move,
                p2_move,
                result,
                timestamp,
                p1_time_ms,
                p2_time_ms
            ]

            df.to_csv(FILE_NAME, index=False)
            game_id += 1

    except KeyboardInterrupt:
        print(f"\nFin del juego, guardando datos en {FILE_NAME}.")


if __name__ == "__main__":
    main()
