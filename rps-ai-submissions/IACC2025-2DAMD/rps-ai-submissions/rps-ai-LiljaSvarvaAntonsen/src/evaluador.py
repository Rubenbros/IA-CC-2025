"""
RPSAI - Evaluador de Winrate
============================
Evalua el rendimiento de mi modelo de IA jugando partidas contra un humano y calculando el winrate.
"""

import sys
from pathlib import Path

# Permitir importar modelo.py desde src
sys.path.insert(0, str(Path(__file__).parent))
from modelo import JugadorIA, GANA_A

# =========================
# MAPEO DE ENTRADAS
# =========================
ENTRADA_A_JUGADA = {
    "1": "piedra", "2": "papel", "3": "tijera",
    "p": "piedra", "a": "papel", "t": "tijera",
    "piedra": "piedra", "papel": "papel", "tijera": "tijera"
}

# Conversion SOLO para el modelo
HUMANO_A_MODELO = {"piedra": "R", "papel": "P", "tijera": "S"}
MODELO_A_TEXTO = {"R": "piedra", "P": "papel", "S": "tijera"}
SIMBOLOS = {"piedra": "P", "papel": "A", "tijera": "T"}

# =========================
# TABLA DE NOTAS
# =========================
def obtener_nota(winrate: float) -> tuple:
    """Calcula la nota basada en el winrate. Returns: (nota, mensaje_especial)"""
    if winrate >= 55: return 10, "BONUS: Fallos en examen no restan!"
    elif winrate >= 50: return 10, None
    elif winrate >= 49: return 9, None
    elif winrate >= 48: return 8, None
    elif winrate >= 46: return 7, None
    elif winrate >= 44: return 6, None
    elif winrate >= 42: return 5, None
    elif winrate >= 40: return 4, None
    elif winrate >= 39: return 3, None
    elif winrate >= 37: return 2, None
    elif winrate >= 35: return 1, None
    else: return 0, None

# =========================
# FUNCIONES AUXILIARES
# =========================
def obtener_resultado(jugada_ia: str, jugada_humano: str) -> str:
    """Resultado desde perspectiva de la IA."""
    if jugada_ia == jugada_humano:
        return "empate"
    elif GANA_A[jugada_ia] == jugada_humano:
        return "victoria"
    else:
        return "derrota"

def leer_jugada_humano() -> str:
    """Lee la jugada del humano."""
    while True:
        print("\n[1/p] Piedra [2/a] Papel [3/t] Tijera")
        entrada = input("Tu jugada: ").lower().strip()
        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]
        print("Jugada no valida. Intenta de nuevo.")

def mostrar_ronda(ronda: int, jugada_ia: str, jugada_humano_txt: str, resultado: str):
    """Muestra el resultado de una ronda."""
    ia_txt = MODELO_A_TEXTO[jugada_ia]
    ia_simbolo = SIMBOLOS[ia_txt]
    print(f"\n--- Ronda {ronda} ---")
    print(f"Tu: {SIMBOLOS[jugada_humano_txt]} ({jugada_humano_txt})")
    print(f"IA: {ia_simbolo} ({ia_txt})")
    if resultado == "victoria":
        print(">>> IA GANA <<<")
    elif resultado == "derrota":
        print(">>> Tú ganas <<<")
    else:
        print(">>> Empate <<<")

def mostrar_progreso(victorias: int, derrotas: int, empates: int, total: int):
    """Muestra el progreso tras cada ronda."""
    jugadas = victorias + derrotas + empates
    restantes = total - jugadas
    decisivas = victorias + derrotas
    winrate = (victorias / decisivas * 100) if decisivas > 0 else 0
    print(
        f"\n[Progreso: {jugadas}/{total}] "
        f"IA: {victorias}V-{derrotas}D-{empates}E "
        f"(Winrate: {winrate:.1f}%) | Quedan: {restantes}"
    )

# =========================
# EVALUACION
# =========================
def evaluar(num_rondas: int = 50):
    print("=" * 60)
    print(" RPSAI - EVALUACION DE WINRATE")
    print("=" * 60)
    print(f"\nSe jugaran {num_rondas} rondas contra mí modelo de IA.")
    print("Juega de forma natural, como lo harias normalmente.\n")

    ia = JugadorIA()
    if ia.modelo is None:
        print("[!] ADVERTENCIA: No se cargo ningun modelo.")
        print("[!] La IA jugara de forma ALEATORIA.")
        print("[!] Entrena tu modelo primero con: python src/modelo.py\n")

    input("Presiona ENTER para comenzar la evaluacion...")

    victorias, derrotas, empates = 0, 0, 0

    for ronda in range(1, num_rondas + 1):
        jugada_ia = ia.decidir_jugada()
        jugada_humano_txt = leer_jugada_humano()
        jugada_humano = HUMANO_A_MODELO[jugada_humano_txt]

        resultado = obtener_resultado(jugada_ia, jugada_humano)
        mostrar_ronda(ronda, jugada_ia, jugada_humano_txt, resultado)

        ia.registrar_ronda(jugada_humano, jugada_ia)

        if resultado == "victoria":
            victorias += 1
        elif resultado == "derrota":
            derrotas += 1
        else:
            empates += 1

        mostrar_progreso(victorias, derrotas, empates, num_rondas)

    # =========================
    # RESULTADOS FINALES
    # =========================
    print("\n" + "=" * 60)
    print(" RESULTADOS FINALES")
    print("=" * 60)

    total_decisivas = victorias + derrotas
    winrate = (victorias / total_decisivas * 100) if total_decisivas > 0 else 0

    print(f"\nRondas jugadas: {num_rondas}")
    print(f"Victorias IA: {victorias}")
    print(f"Derrotas IA: {derrotas}")
    print(f"Empates: {empates}")
    print(f"\nWINRATE DE LA IA: {winrate:.1f}%")

    nota, bonus = obtener_nota(winrate)
    print(f"\n{'=' * 60}")
    print(f" NOTA MODO 1 (Winrate): {nota}/10")
    if bonus:
        print(f" {bonus}")
    print(f"{'=' * 60}")

    # Tabla de referencia
    print("\nTabla de referencia:")
    print(" 33%- = 0 | 35% = 1 | 37% = 2 | 39% = 3")
    print(" 40% = 4 | 42% = 5 | 44% = 6 | 46% = 7")
    print(" 48% = 8 | 49% = 9 | 50%+ = 10 | 55%+ = BONUS")

# =========================
# MAIN
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evalua el winrate de mí modelo de IA")
    parser.add_argument("-n", "--rondas", type=int, default=50, help="Numero de rondas a jugar (default: 50)")
    args = parser.parse_args()
    evaluar(args.rondas)

if __name__ == "__main__":
    main()
