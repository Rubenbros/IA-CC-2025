import csv
import os
import time
from datetime import datetime


def crear_directorio():
    """Crea el directorio data/ si no existe"""
    if not os.path.exists('data'):
        os.makedirs('data')


def guardar_jugada(numero_ronda, jugada_j1, jugada_j2, timestamp, tiempo_reaccion, sesion, resultado):
    """Guarda una jugada en el CSV"""
    file_exists = os.path.isfile('../data/partidas.csv')

    with open('../data/partidas.csv', 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['numero_ronda', 'jugada_j1', 'jugada_j2', 'timestamp',
                      'tiempo_reaccion_ms', 'sesion', 'resultado']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'numero_ronda': numero_ronda,
            'jugada_j1': jugada_j1,
            'jugada_j2': jugada_j2,
            'timestamp': timestamp,
            'tiempo_reaccion_ms': tiempo_reaccion,
            'sesion': sesion,
            'resultado': resultado
        })


def determinar_ganador(jugada1, jugada2):
    """Determina el resultado de la ronda"""
    if jugada1 == jugada2:
        return 'empate'

    ganadores = {
        'piedra': 'tijera',
        'papel': 'piedra',
        'tijera': 'papel'
    }

    if ganadores[jugada1] == jugada2:
        return 'victoria'
    else:
        return 'derrota'


def jugar_partida():
    """Juega una partida completa"""
    crear_directorio()

    print("\n" + "=" * 50)
    print("🎮 RECOLECTOR DE DATOS - PIEDRA, PAPEL O TIJERA")
    print("=" * 50)

    sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📊 ID de Sesión: {sesion}")

    nombre_oponente = input("\n👤 Nombre del oponente (deja vacío para 'Oponente'): ").strip()
    if not nombre_oponente:
        nombre_oponente = "Oponente"

    print(f"\n¡Perfecto! Vas a jugar contra: {nombre_oponente}")
    print("\nInstrucciones:")
    print("  - Tú eres J1 (Jugador 1)")
    print(f"  - {nombre_oponente} es J2 (Jugador 2)")
    print("  - Escribe: piedra, papel o tijera")
    print("  - Escribe 'fin' para terminar la sesión\n")

    ronda = 1
    victorias_j1 = 0
    victorias_j2 = 0
    empates = 0

    while True:
        print(f"\n--- RONDA {ronda} ---")
        print(f"📊 Marcador: Tú {victorias_j1} - {victorias_j2} {nombre_oponente} (Empates: {empates})")

        # Jugada J1 (usuario)
        tiempo_inicio = time.time()
        jugada_j1 = input("Tu jugada (piedra/papel/tijera): ").lower().strip()
        tiempo_reaccion = int((time.time() - tiempo_inicio) * 1000)

        if jugada_j1 == 'fin':
            print(f"\n✅ Sesión terminada. Total de rondas: {ronda - 1}")
            print(f"📊 Resultado final: Tú {victorias_j1} - {victorias_j2} {nombre_oponente} (Empates: {empates})")
            break

        if jugada_j1 not in ['piedra', 'papel', 'tijera']:
            print("❌ Jugada inválida. Usa: piedra, papel o tijera")
            continue

        # Jugada J2 (oponente)
        jugada_j2 = input(f"Jugada de {nombre_oponente} (piedra/papel/tijera): ").lower().strip()

        if jugada_j2 not in ['piedra', 'papel', 'tijera']:
            print("❌ Jugada inválida. Usa: piedra, papel o tijera")
            continue

        # Determinar resultado
        resultado = determinar_ganador(jugada_j1, jugada_j2)

        # Actualizar marcador
        if resultado == 'victoria':
            victorias_j1 += 1
            print(f"✅ ¡Ganaste! {jugada_j1} vence a {jugada_j2}")
        elif resultado == 'derrota':
            victorias_j2 += 1
            print(f"❌ Perdiste. {jugada_j2} vence a {jugada_j1}")
        else:
            empates += 1
            print(f"🤝 Empate. Ambos jugaron {jugada_j1}")

        # Guardar en CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guardar_jugada(ronda, jugada_j1, jugada_j2, timestamp, tiempo_reaccion, sesion, resultado)

        ronda += 1

    print(f"\n💾 Datos guardados en: data/partidas.csv")
    print("=" * 50 + "\n")


def modo_rapido():
    """Modo rápido para ingresar múltiples jugadas rápidamente"""
    crear_directorio()

    print("\n" + "=" * 50)
    print("⚡ MODO RÁPIDO - Ingreso de Datos")
    print("=" * 50)

    sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📊 ID de Sesión: {sesion}")

    nombre_oponente = input("\n👤 Nombre del oponente: ").strip()
    if not nombre_oponente:
        nombre_oponente = "Oponente"

    print("\nFormato: <tu_jugada> <jugada_oponente>")
    print("Ejemplo: piedra papel")
    print("Escribe 'fin' para terminar\n")

    ronda = 1

    while True:
        entrada = input(f"Ronda {ronda}: ").lower().strip()

        if entrada == 'fin':
            print(f"\n✅ Datos guardados. Total: {ronda - 1} rondas")
            break

        partes = entrada.split()
        if len(partes) != 2:
            print("❌ Formato incorrecto. Usa: <tu_jugada> <jugada_oponente>")
            continue

        jugada_j1, jugada_j2 = partes

        if jugada_j1 not in ['piedra', 'papel', 'tijera'] or jugada_j2 not in ['piedra', 'papel', 'tijera']:
            print("❌ Jugadas inválidas")
            continue

        resultado = determinar_ganador(jugada_j1, jugada_j2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        guardar_jugada(ronda, jugada_j1, jugada_j2, timestamp, 0, sesion, resultado)

        ronda += 1

    print(f"\n💾 Datos guardados en: data/partidas.csv\n")


if __name__ == "__main__":
    print("\n🎯 Selecciona el modo:")
    print("1. Modo normal (jugadas en tiempo real)")
    print("2. Modo rápido (ingreso masivo de datos)")

    modo = input("\nElige (1/2): ").strip()

    if modo == "2":
        modo_rapido()
    else:
        jugar_partida()