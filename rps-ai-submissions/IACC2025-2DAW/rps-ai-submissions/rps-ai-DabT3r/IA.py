import csv
import keyboard

# Jugador 1 (izquierda): a = piedra, s = papel, d = tijera
# Jugador 2 (derecha): 1 = piedra, 2 = papel, 3 = tijera

key_to_move_p1 = {
    'a': 'piedra',
    's': 'papel',
    'd': 'tijera'
}

key_to_move_p2 = {
    '1': 'piedra',
    '2': 'papel',
    '3': 'tijera'
}


def wait_for_moves():
    """
    Espera hasta que ambos jugadores pulsen una tecla válida.
    El primero que pulse en cada grupo queda registrado.
    """

    p1_move = None
    p2_move = None

    print("Jugadores preparados... (Jugador1: a/s/d — Jugador2: 1/2/3)")
    print("Pulsen a la vez para mayor simultaneidad.")

    while p1_move is None or p2_move is None:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name in key_to_move_p1 and p1_move is None:
                p1_move = key_to_move_p1[event.name]
            elif event.name in key_to_move_p2 and p2_move is None:
                p2_move = key_to_move_p2[event.name]

    return p1_move, p2_move


def determine_winner(p1, p2):
    if p1 == p2:
        return "empate"
    if (
        (p1 == "piedra" and p2 == "tijera") or
        (p1 == "papel" and p2 == "piedra") or
        (p1 == "tijera" and p2 == "papel")
    ):
        return "jugador1"
    return "jugador2"


def save_data(filename, row):
    header = ["ronda", "player1_move", "player2_move", "winner"]
    try:
        with open(filename, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except FileExistsError:
        pass

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def play_and_collect(filename="Datos.csv"):
    ronda = 0
    print("Sistema de recopilación de datos - Piedra, Papel o Tijera con teclas simultáneas")
    print("Jugador1: a,s,d — Jugador2: 1,2,3")
    print("Pulsa ESC para salir.")

    while True:
        if keyboard.is_pressed('esc'):
            print("Saliendo... ¡Gracias por jugar!")
            break

        p1_move, p2_move = wait_for_moves()

        winner = determine_winner(p1_move, p2_move)
        print(f"Jugador1: {p1_move}  |  Jugador2: {p2_move}  →  Resultado: {winner}")
        ronda += 1

        save_data(filename, [ronda, p1_move, p2_move, winner])


if __name__ == "__main__":
    play_and_collect()