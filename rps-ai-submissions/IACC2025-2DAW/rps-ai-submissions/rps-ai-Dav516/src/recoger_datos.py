# src/play_interactive.py
from modelo import JugadorIA
import csv
from pathlib import Path
import pandas as pd

# Archivo donde se guardarán las partidas
CSV_FILE = Path(__file__).parent / "../data/partidas.csv"

def obtener_ultima_ronda():
    """Devuelve el número de la última ronda registrada en el CSV"""
    if not CSV_FILE.exists():
        return 0  # no hay rondas guardadas
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Comprobar que existe la columna esperada
        if "numero_ronda" not in reader.fieldnames:
            return 0
        filas = list(reader)
        if not filas:
            return 0
        return max(int(fila["numero_ronda"]) for fila in filas)

def guardar_ronda(ronda, user, ia):
    """Guarda la ronda en partidas.csv, creando cabecera si no existe"""
    existe = CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            # Cabecera EXACTA que espera modelo.py
            writer.writerow(["numero_ronda", "jugada_j1", "jugada_j2"])
        writer.writerow([ronda, user, ia])

def main():
    ia = JugadorIA()  # carga modelo desde models/modelo_entrenado.pkl
    ultima_ronda = obtener_ultima_ronda()
    ronda = ultima_ronda + 1

    print(f"Modelo cargado.")
    print(f"Juega contra la IA. Empiezas desde la ronda {ronda}. Escribe 'salir' para terminar.")

    while True:
        user = input(f"[Ronda {ronda}] Tu jugada (piedra/papel/tijera): ").strip().lower()
        if user == "salir":
            print("¡Hasta luego!")
            break
        if user not in ["piedra", "papel", "tijera"]:
            print("Entrada inválida. Intenta de nuevo.")
            continue

        # La IA decide su jugada usando DataFrame con columnas correctas
        jug_ia = ia.decidir_jugada()
        print(f"IA juega: {jug_ia}")

        # registrar la ronda en memoria para la IA
        ia.registrar_ronda(user, jug_ia)

        # guardar la ronda en CSV
        guardar_ronda(ronda, user, jug_ia)

        # resultado simple
        if user == jug_ia:
            print("Empate")
        elif (user == "piedra" and jug_ia == "tijera") or \
             (user == "tijera" and jug_ia == "papel") or \
             (user == "papel" and jug_ia == "piedra"):
            print("¡Tú ganas!")
        else:
            print("La IA gana.")

        ronda += 1

if __name__ == "__main__":
    main()
