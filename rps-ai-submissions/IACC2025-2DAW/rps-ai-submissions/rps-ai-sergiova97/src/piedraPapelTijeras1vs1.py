import csv
import os
import keyboard
import time

OPTIONS_1 = {'a': 'Piedra', 's': 'Papel', 'd': 'Tijeras'}
OPTIONS_2 = {'1': 'Piedra', '2': 'Papel', '3': 'Tijeras'}

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir))
FILE_PATH = os.path.join(root_dir, 'matches')
os.makedirs(FILE_PATH, exist_ok=True)


GENERAL_FILE = os.path.join(FILE_PATH, 'general1vs1.csv')

def save_match(play, name1, name2, choose1, choose2, result, time1, time2):
    """Saves the match data in file"""
    headers = ['ronda', 'jugador1', 'jugador2', 'jugada1', 'jugada2', 'ganador', 'tiempo1', 'tiempo2']

    if not os.path.exists(GENERAL_FILE):
        with open(GENERAL_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    row = [play, name1, name2, choose1, choose2, result, time1, time2]
    with open(GENERAL_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def determine_winner(player1, player2):
    """Returns the winner of the game"""

    if player1 == player2:
        return 0
    elif (player1 == 'Piedra' and player2 == 'Tijeras') or \
         (player1 == 'Papel' and player2 == 'Piedra') or \
         (player1 == 'Tijeras' and player2 == 'Papel'):
        return 1
    else:
        return 2

def determine_winner_name(name1, choose1, name2, choose2):
    """Returns the name of the winner of the game"""
    result = determine_winner(choose1, choose2)

    if result == 0:
        return "Empate"
    elif result == 1:
        return name1
    elif result == 2:
        return name2
    else:
        return 'Error inesperado'

def play():
    name1 = input('Ingresa el nombre del Jugador1: ').strip()
    name2 = input('Ingresa el nombre del Jugador2: ').strip()

    print('\nJugador 1 teclas: a=Piedra, s=Papel, d=Tijeras')
    print('Jugador 2 teclas: 1=Piedra, 2=Papel, 3=Tijeras')
    print('Pulsad las dos teclas al mismo tiempo. [Esc] para salir\n')

    i = 0

    while True:
        choose_j1 = None
        choose_j2 = None
        time_j1 = None
        time_j2 = None

        start_time = time.time()

        while choose_j1 is None or choose_j2 is None:
            event = keyboard.read_event()
            current_time = time.time()

            if event.event_type == keyboard.KEY_DOWN:
                key = event.name.lower()
                if key == 'esc':
                    print('¡Gracias por jugar!')
                    return
                if key in OPTIONS_1 and choose_j1 is None:
                    choose_j1 = OPTIONS_1[key]
                    time_j1 = current_time - start_time
                if key in OPTIONS_2 and choose_j2 is None:
                    choose_j2 = OPTIONS_2[key]
                    time_j2 = current_time - start_time

        i += 1
        result_name = determine_winner_name(name1, choose_j1, name2, choose_j2)
        save_match(i, name1, name2, choose_j1, choose_j2, result_name, time_j1, time_j2)

        print(f"{name1} eligió: {choose_j1}")
        print(f"{name2} eligió: {choose_j2}")
        print(f"Ganador: {result_name}")

if __name__ == '__main__':
    play()