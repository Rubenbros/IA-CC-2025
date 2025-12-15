"""
RPSAI - Evaluador de Winrate (MEJORADO PARA IA INTELIGENTE)
===========================================================

Este evaluador funciona igual que el original pero añade:
- Información de si la IA ha usado patrones del rival
- Información de si la IA ha usado ML o modo seguro
- Estadísticas del estilo del rival al final
"""
import sys
from pathlib import Path

# Añadir carpeta src al path
sys.path.insert(0, str(Path(__file__).parent))

from modelo import (
    JugadorIA, JUGADA_A_NUM, NUM_A_JUGADA, GANA_A
)

# ------------------------------------------
# MAPEO DE ENTRADA DEL HUMANO
# ------------------------------------------
ENTRADA_A_JUGADA = {
    "1": "piedra",
    "2": "papel",
    "3": "tijera",
    "p": "piedra",
    "a": "papel",
    "t": "tijera",
    "piedra": "piedra",
    "papel": "papel",
    "tijera": "tijera"
}

def leer_jugada_humano():
    while True:
        print("\n[1/p] Piedra [2/a] Papel [3/t] Tijera")
        entrada = input("Tu jugada: ").lower().strip()
        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]
        print("Entrada no válida.")

# ------------------------------------------
# RESULTADO
# ------------------------------------------
def obtener_resultado(jugada_ia, jugada_humano):
    if jugada_ia == jugada_humano:
        return "empate"
    elif GANA_A[jugada_ia] == jugada_humano:
        return "victoria"
    else:
        return "derrota"


# ---------- SOLO CAMBIO LO QUE SE PRINTA ----------
def mostrar_ronda(ronda, jug_hum, jug_ia, resultado, modo):
    simbolos = {"piedra": "P", "papel": "A", "tijera": "T"}

    print(f"\n--- Ronda {ronda} ---")
    print(f"Tu: {simbolos[jug_hum]} ({jug_hum})")
    print(f"IA: {simbolos[jug_ia]} ({jug_ia})")

    if resultado == "victoria":
        print(">>> IA GANA <<<")
    elif resultado == "derrota":
        print(">>> Tu ganas <<<")
    else:
        print(">>> Empate <<<")




def mostrar_progreso(v, d, e, total):
    jugadas = v + d + e
    restantes = total - jugadas

    if v + d > 0:
        winrate = v / (v + d) * 100
        print(f"\n[Progreso: {jugadas}/{total}] "
              f"IA: {v}V-{d}D-{e}E "
              f"(Winrate: {winrate:.1f}%) | Quedan: {restantes}")


# ------------------------------------------
# EVALUACIÓN
# ------------------------------------------
def evaluar(num_rondas=50):

    print("="*60)
    print(" RPSAI - EVALUACION DE WINRATE")
    print("="*60)
    print(f"\nSe jugarán {num_rondas} rondas contra tu modelo.\n")

    try:
        ia = JugadorIA()
        if ia.modelo is None:
            print("[!] ADVERTENCIA: No se cargó ningún modelo.")
            print("[!] La IA jugará ALEATORIA.\n")
    except:
        ia = JugadorIA()
        print("[!] Error cargando modelo. Modo aleatorio.\n")

    input("Pulsa ENTER para empezar...")

    v = d = e = 0

    for ronda in range(1, num_rondas + 1):

        jugada_humano = leer_jugada_humano()

        modo = "ML / patrón"
        jugada_ia = ia.decidir_jugada()

        if ia.racha_perdidas >= 2:
            modo = "MODO CAOS"
        if len(ia.historial) < 3:
            modo = "Semialeatorio"

        resultado = obtener_resultado(jugada_ia, jugada_humano)

        # Mostrar ronda en formato BÁSICO
        mostrar_ronda(ronda, jugada_humano, jugada_ia, resultado, modo)

        ia.registrar_ronda(jugada_humano, jugada_ia)

        if resultado == "victoria":
            v += 1
        elif resultado == "derrota":
            d += 1
        else:
            e += 1

        mostrar_progreso(v, d, e, num_rondas)

    # ------------------------------------------
    # RESULTADOS FINALES (FORMATO SIMPLE)
    # ------------------------------------------
    print("\n" + "="*60)
    print(" RESULTADOS FINALES")
    print("="*60)

    decisivas = v + d
    winrate = (v / decisivas * 100) if decisivas > 0 else 0

    print(f"Rondas jugadas: {num_rondas}")
    print(f"Victorias IA: {v}")
    print(f"Derrotas IA: {d}")
    print(f"Empates: {e}")
    print(f"WINRATE DE LA IA: {winrate:.1f}%")



    print("\nFIN DE LA EVALUACIÓN.")
    print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--rondas", type=int, default=50)
    args = parser.parse_args()
    evaluar(args.rondas)

if __name__ == "__main__":
    main()
