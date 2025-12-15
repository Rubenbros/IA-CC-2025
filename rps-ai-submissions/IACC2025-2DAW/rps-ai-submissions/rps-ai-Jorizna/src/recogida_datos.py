import time
import pandas as pd
import os

# Ruta absoluta a la carpeta del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "partidas.csv")

# Si keyboard no está instalado, lo instalamos
try:
    import keyboard
except ImportError:
    os.system(f"{os.sys.executable} -m pip install keyboard")
    import keyboard

# Diccionario de teclas → jugada
jugador1_teclas = {'a': 'piedra', 's': 'papel', 'd': 'tijera'}
jugador2_teclas = {'b': 'piedra', 'n': 'papel', 'm': 'tijera'}

cont = 1
seguirB = True

# Función para determinar el ganador
def ganador(j1, j2):
    if j1 == j2:
        return "empate"

    reglas = {
        "piedra": "tijera",
        "tijera": "papel",
        "papel": "piedra"
    }

    if reglas[j1] == j2:
        return "gana_jugador1"
    else:
        return "gana_jugador2"

# Obtener nombres
Jugador1 = input("Introduce el nombre del jugador 1: ")
Jugador2 = input("Introduce el nombre del jugador 2: ")

print("Jugador1 usa: A = piedra | S = papel | D = tijera")
print("Jugador2 usa: B = piedra | N = papel | M = tijera")
print("\nCuando estéis listos, pulsad vuestra tecla a la vez.\n")

# Crear CSV si no existe
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=[
        "Num_Ronda", "Jugador1", "jugada_j1", "tiempo_j1",
        "Jugador2", "jugada_j2", "tiempo_j2", "resultado"
    ])
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

# Bucle principal
while seguirB:
    print("\n--- NUEVA RONDA ---")
    print("Esperando jugadas...")

    ronda_inicio = time.time()

    jugada_jugador1 = None
    jugada_jugador2 = None
    start_jugador1 = None
    start_jugador2 = None

    # Esperamos jugadas simultáneas
    while jugada_jugador1 is None or jugada_jugador2 is None:

        # Jugador 1
        for k in jugador1_teclas:
            if keyboard.is_pressed(k) and jugada_jugador1 is None:
                jugada_jugador1 = jugador1_teclas[k]
                start_jugador1 = time.time() - ronda_inicio
                print(f"J1 jugó: {jugada_jugador1} (tiempo: {start_jugador1:.3f}s)")

        # Jugador 2
        for k in jugador2_teclas:
            if keyboard.is_pressed(k) and jugada_jugador2 is None:
                jugada_jugador2 = jugador2_teclas[k]
                start_jugador2 = time.time() - ronda_inicio
                print(f"J2 jugó: {jugada_jugador2} (tiempo: {start_jugador2:.3f}s)")

    # Resultado
    resultado = ganador(jugada_jugador1, jugada_jugador2)
    print(f"\nResultado: {resultado}")

    # Nueva fila
    nueva_fila = {
        "Num_Ronda": cont,
        "Jugador1": Jugador1,
        "jugada_j1": jugada_jugador1,
        "tiempo_j1": round(start_jugador1, 3),
        "Jugador2": Jugador2,
        "jugada_j2": jugada_jugador2,
        "tiempo_j2": round(start_jugador2, 3),
        "resultado": resultado
    }

    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)

    # Guardar CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Datos guardados en {CSV_PATH}")

    # Preguntar si seguir
    seguir = input("\n¿Jugar otra vez? (s/n): ").lower()
    if seguir != "n":
        cont += 1
    else:
        print("Juego terminado.")
        seguirB = False