"""
RPSAI - Evaluador de Winrate
============================

Este script evalúa el rendimiento de tu modelo de IA
jugando partidas contra un humano y calculando el winrate.

IMPORTANTE: El script intenta cargar el modelo entrenado. Si no lo encuentra,
la IA jugará de forma ALEATORIA, pero seguirá recopilando datos.
"""

import sys
import csv
import os
from pathlib import Path


# La ruta ajustada para tu estructura de carpetas (evaluador.py en src/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "partidas.csv"

# Importamos las clases y constantes necesarias del modelo
try:
    from modelo import JugadorIA, GANA_A
except ImportError:
    print("Error de importación: Asegúrate de que 'modelo.py' existe y la clase JugadorIA es accesible.")
    sys.exit(1)

# Mapeo de entrada a jugada
ENTRADA_A_JUGADA = {
    "1": "piedra", "2": "papel", "3": "tijera",
    "p": "piedra", "a": "papel", "t": "tijera",
    "piedra": "piedra", "papel": "papel", "tijera": "tijera"
}



def guardar_jugada(ronda: int, jugada_ia: str, jugada_humano: str, inicial=False):
    """
    Guarda una ronda jugada en el archivo partidas.csv.
    """
    # 1. Asegurar que la carpeta 'data' existe
    os.makedirs(RUTA_DATOS.parent, exist_ok=True)

    # 2. Abrir el archivo en modo 'a' (append/añadir)
    try:
        with open(RUTA_DATOS, 'a', newline='') as f:
            writer = csv.writer(f)

            if inicial:
                if not RUTA_DATOS.exists() or os.path.getsize(RUTA_DATOS) == 0:
                    writer.writerow(["numero_ronda", "jugada_j1", "jugada_j2"])
                    print(f"[LOG] Archivo de datos inicializado en: {RUTA_DATOS}")
            else:
                writer.writerow([ronda, jugada_ia, jugada_humano])

    except Exception as e:
        print(f"[ERROR] No se pudo escribir en el archivo CSV: {e}")


def obtener_nota(winrate: float) -> tuple:
    """Calcula la nota basada en el winrate."""
    if winrate >= 55:
        return 10, "BONUS: Fallos en examen no restan!"
    elif winrate >= 50:
        return 10, None
    elif winrate >= 49:
        return 9, None
    elif winrate >= 48:
        return 8, None
    elif winrate >= 46:
        return 7, None
    elif winrate >= 44:
        return 6, None
    elif winrate >= 42:
        return 5, None
    elif winrate >= 40:
        return 4, None
    elif winrate >= 39:
        return 3, None
    elif winrate >= 37:
        return 2, None
    elif winrate >= 35:
        return 1, None
    else:
        return 0, None


def obtener_resultado(jugada_ia: str, jugada_humano: str) -> str:
    """Obtiene el resultado desde la perspectiva de la IA (jugada_ia)."""
    if jugada_ia == jugada_humano:
        return "empate"
    elif GANA_A.get(jugada_ia) == jugada_humano:
        return "victoria"
    else:
        return "derrota"


def leer_jugada_rival() -> str:
    """Lee la jugada del rival (el humano que juega contra la IA)."""
    while True:
        print("\n[1/p] Piedra  [2/a] Papel  [3/t] Tijera")
        # SOLO pedimos la jugada del rival
        entrada = input("Jugada del Rival (Tu Amigo): ").lower().strip()

        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]

        if entrada == "salir":
            return "salir"

        print("Jugada no valida. Intenta de nuevo.")


def mostrar_ronda(ronda: int, jugada_ia: str, jugada_humano: str, resultado: str):
    """Muestra el resultado de una ronda."""
    simbolos = {"piedra": "P", "papel": "A", "tijera": "T"}

    print(f"\n--- Ronda {ronda} ---")
    print(f"IA (Mi jugada): {simbolos.get(jugada_ia, '?')} ({jugada_ia})")
    print(f"Rival: {simbolos.get(jugada_humano, '?')} ({jugada_humano})")

    if resultado == "victoria":
        print(">>> IA GANA <<<")
    elif resultado == "derrota":
        print(">>> Rival gana <<<")
    else:
        print(">>> Empate <<<")


def mostrar_progreso(victorias: int, derrotas: int, empates: int, total: int):
    """Muestra el progreso actual."""
    jugadas = victorias + derrotas + empates
    total_decisivas = victorias + derrotas
    restantes = total - jugadas

    winrate_actual = 0
    if total_decisivas > 0:
        winrate_actual = victorias / total_decisivas * 100

    print(f"\n[Progreso: {jugadas}/{total}] "
          f"IA: {victorias}V-{derrotas}D-{empates}E "
          f"(Winrate: {winrate_actual:.1f}%) "
          f"| Quedan: {restantes}")



def evaluar(num_rondas: int = 50):
    """
    Ejecuta la evaluación del modelo con predicción automática.
    """
    print("=" * 60)
    print("   RPSAI - EVALUACIÓN DE WINRATE (MODO PREDICCIÓN)")
    print("=" * 60)
    print(f"\nSe jugarán {num_rondas} rondas contra la IA (escribe 'salir' para terminar antes).")

    # Intentar cargar el modelo
    ia = None
    try:
        ia = JugadorIA()
        if ia.modelo is None:
            print("[!] ADVERTENCIA: No se cargó ningún modelo. La IA jugará ALEATORIA.")
            print("[!] Entrena tu modelo primero con: python src/modelo.py\n")
    except Exception as e:
        print(f"[!] Error al inicializar JugadorIA: {e}")
        ia = JugadorIA()  # Aseguramos que la clase existe para que no falle el resto del código

    # 1. Inicializa el archivo CSV con la cabecera si es necesario
    guardar_jugada(0, "", "", inicial=True)

    input("Presiona ENTER para comenzar la evaluación...")

    victorias = 0
    derrotas = 0
    empates = 0
    rondas_jugadas = 0

    for ronda in range(1, num_rondas + 1):
        rondas_jugadas += 1

        # 1. La IA PREDICE Y DECIDE su jugada automáticamente
        jugada_ia = ia.decidir_jugada()

        # 2. PEDIR JUGADA DEL RIVAL (SOLO UNA ENTRADA)
        jugada_humano = leer_jugada_rival()

        if jugada_humano == "salir":
            rondas_jugadas -= 1
            break

        # 3. Determinar resultado (desde perspectiva IA)
        resultado = obtener_resultado(jugada_ia, jugada_humano)

        # 4. Mostrar resultado
        mostrar_ronda(rondas_jugadas, jugada_ia, jugada_humano, resultado)

        # 5. Registrar en el historial de la IA (para la siguiente predicción)
        ia.registrar_ronda(jugada_ia, jugada_humano)

        # 6. Guardar la jugada en el CSV
        guardar_jugada(rondas_jugadas, jugada_ia, jugada_humano)

        # 7. Actualizar contadores
        if resultado == "victoria":
            victorias += 1
        elif resultado == "derrota":
            derrotas += 1
        else:
            empates += 1

        # 8. Mostrar progreso
        mostrar_progreso(victorias, derrotas, empates, num_rondas)

    # Resultados finales

    print("\n" + "=" * 60)
    print("   RESULTADOS FINALES")
    print("=" * 60)

    total_decisivas = victorias + derrotas
    rondas_totales = victorias + derrotas + empates

    winrate = 0
    if total_decisivas > 0:
        winrate = victorias / total_decisivas * 100

    print(f"\nRondas jugadas: {rondas_totales}")
    print(f"Victorias IA: {victorias}")
    print(f"Derrotas IA: {derrotas}")
    print(f"Empates: {empates}")
    print(f"\nWINRATE DE LA IA: {winrate:.1f}% (Calculado sobre decisivas)")

    nota, bonus = obtener_nota(winrate)
    print(f"\n{'=' * 60}")
    print(f"   NOTA MODO 1 (Winrate): {nota}/10")
    if bonus:
        print(f"   {bonus}")
    print(f"{'=' * 60}")

    print(f"\n[LOG] ¡{rondas_totales} rondas añadidas/actualizadas en {RUTA_DATOS}!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Evalua el winrate de tu modelo de IA")
    parser.add_argument("-n", "--rondas", type=int, default=50,
                        help="Numero de rondas a jugar (default: 50)")
    args = parser.parse_args()

    evaluar(args.rondas)


if __name__ == "__main__":
    main()
