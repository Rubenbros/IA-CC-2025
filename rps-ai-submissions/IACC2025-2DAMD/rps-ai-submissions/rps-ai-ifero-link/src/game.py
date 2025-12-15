import pygame
import csv
import os
import time

# -------------------------------------------------------
# Dic Jugador1: A:Piedra - S:Papel - D:Tijera
# -------------------------------------------------------
jugadas_j1 = {
    pygame.K_a: "piedra",
    pygame.K_s: "papel",
    pygame.K_d: "tijera",
}
# -------------------------------------------------------
# Dic Jugador2: J:Piedra - K:Papel - L:Tijera
# -------------------------------------------------------
jugadas_j2 = {
    pygame.K_j: "piedra",
    pygame.K_k: "papel",
    pygame.K_l: "tijera",
}
# -------------------------------------------------------
# Reglas del juego
# -------------------------------------------------------
reglas = {
    "piedra": "tijera",
    "papel": "piedra",
    "tijera": "papel"
}
# -------------------------------------------------------
# Compara la jugada de los dos usuarios y devuelve un string con la quien ha ganado
# -------------------------------------------------------
def determinar_resultado(j1, j2):
    if j1 == j2:
        return "empate"
    if reglas[j1] == j2:
        return "gana_j1"
    return "gana_j2"


# -------------------------------------------------------
# Juego principal
# -------------------------------------------------------
def juego():

    # Crear carpeta para el CSV
    carpeta = "data_game"
    os.makedirs(carpeta, exist_ok=True)

    #Lo creamos en otra carperta para que no se mezcle con los datos finales
    archivo_csv = os.path.join(carpeta, "partida_ppt_Ana_4.csv")
    campos = ["numero_ronda", "jugada_j1", "jugada_j2", "resultado",
              "cambia_j1", "cambia_j2"]

    rondas = []
    ronda_actual = 1

    # Jugadas anteriores
    jugada_anterior_j1 = None
    jugada_anterior_j2 = None

    # Crear la pantalla de juego con pygame
    pygame.init()
    pantalla = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("Piedra, Papel o Tijera - Simultáneo")

    fuente = pygame.font.Font(None, 36)

    def mostrar_texto(texto):
        pantalla.fill((0, 0, 0))
        txt = fuente.render(texto, True, (255, 255, 255))
        pantalla.blit(txt, (50, 120))
        pygame.display.flip()

    jugando = True

    while jugando:

        for i in range(3, 0, -1):
            mostrar_texto(f"Preparados... {i}")
            time.sleep(1)

        mostrar_texto("¡YA! Mantened la tecla")
        time.sleep(0.3)

        j1 = None
        j2 = None

        tiempo_limite = time.time() + 1.0

        while time.time() < tiempo_limite:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    jugando = False
                    break

            teclas = pygame.key.get_pressed()

            for t, jugada in jugadas_j1.items():
                if teclas[t]:
                    j1 = jugada

            for t, jugada in jugadas_j2.items():
                if teclas[t]:
                    j2 = jugada

            if j1 and j2:
                break

        if not jugando:
            break

        if j1 is None or j2 is None:
            mostrar_texto("No se detectaron ambas jugadas. Repetimos ronda...")
            time.sleep(2)
            continue

        resultado = determinar_resultado(j1, j2)

        # ------------------------------
        # Detectar cambios de jugada
        # ------------------------------
        cambia_j1 = (jugada_anterior_j1 is not None and j1 != jugada_anterior_j1)
        cambia_j2 = (jugada_anterior_j2 is not None and j2 != jugada_anterior_j2)

        # Guardar en CSV
        rondas.append([
            ronda_actual,
            j1,
            j2,
            resultado,
            cambia_j1,
            cambia_j2
        ])

        # Actualizar jugadas anteriores
        jugada_anterior_j1 = j1
        jugada_anterior_j2 = j2

        ronda_actual += 1

        mostrar_texto(f"Resultado: {resultado}")
        time.sleep(2)

    # ------------------------------
    # Escribir los campos que se han recogido.
    # ------------------------------
    with open(archivo_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(campos)
        writer.writerows(rondas)

    pygame.quit()
    print(f"\nCSV generado en: {archivo_csv}")

# Lanzar el juego
if __name__ == "__main__":
    juego()
