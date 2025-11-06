"""
Generador de Datos de Ejemplo para PPT
=======================================

Este script genera un CSV con datos de partidas de Piedra, Papel o Tijera
para practicar entrenamiento de modelos.

PATRONES SIMULADOS:
1. Oponente básico: Prefiere piedra (40%), papel (35%), tijera (25%)
2. Oponente adaptativo: Cambia estrategia según resultados
3. Oponente patrones: Sigue secuencias repetitivas
"""

import pandas as pd
import numpy as np


def generar_oponente_basico(n_rondas=200):
    """
    Oponente con preferencias fijas
    Prefiere piedra > papel > tijera
    """
    np.random.seed(42)

    jugadas_oponente = np.random.choice(
        ['piedra', 'papel', 'tijera'],
        size=n_rondas,
        p=[0.40, 0.35, 0.25]  # Preferencia
    )

    # Jugador con estrategia counter básica
    counter = {'piedra': 'papel', 'papel': 'tijera', 'tijera': 'piedra'}
    jugadas_jugador = []

    for jug_op in jugadas_oponente:
        if np.random.random() < 0.7:  # 70% usa counter
            jugadas_jugador.append(counter[jug_op])
        else:  # 30% aleatorio
            jugadas_jugador.append(np.random.choice(['piedra', 'papel', 'tijera']))

    return jugadas_jugador, jugadas_oponente


def generar_oponente_adaptativo(n_rondas=200):
    """
    Oponente que adapta su estrategia según si gana o pierde
    Si pierde: tiende a cambiar de jugada
    Si gana: tiende a repetir jugada
    """
    np.random.seed(43)

    jugadas_oponente = []
    jugadas_jugador = []

    # Calcular ganador
    def quien_gana(jug1, jug2):
        if jug1 == jug2:
            return 'empate'
        gana = {'piedra': 'tijera', 'papel': 'piedra', 'tijera': 'papel'}
        return 'jug1' if gana[jug1] == jug2 else 'jug2'

    # Primera jugada aleatoria
    jug_op = np.random.choice(['piedra', 'papel', 'tijera'])
    jug_jug = np.random.choice(['piedra', 'papel', 'tijera'])

    jugadas_oponente.append(jug_op)
    jugadas_jugador.append(jug_jug)

    # Siguientes jugadas
    for i in range(1, n_rondas):
        resultado = quien_gana(jug_op, jug_jug)

        if resultado == 'jug2':  # Oponente ganó
            # 60% repite jugada ganadora
            if np.random.random() < 0.6:
                jug_op = jugadas_oponente[-1]
            else:
                jug_op = np.random.choice(['piedra', 'papel', 'tijera'])
        else:  # Oponente perdió o empató
            # 70% cambia de jugada
            if np.random.random() < 0.7:
                opciones = [j for j in ['piedra', 'papel', 'tijera'] if j != jugadas_oponente[-1]]
                jug_op = np.random.choice(opciones)
            else:
                jug_op = np.random.choice(['piedra', 'papel', 'tijera'])

        # Jugador usa estrategia mixta
        counter = {'piedra': 'papel', 'papel': 'tijera', 'tijera': 'piedra'}
        if np.random.random() < 0.65:
            jug_jug = counter[jugadas_oponente[-1]]  # Counter basado en última jugada
        else:
            jug_jug = np.random.choice(['piedra', 'papel', 'tijera'])

        jugadas_oponente.append(jug_op)
        jugadas_jugador.append(jug_jug)

    return jugadas_jugador, jugadas_oponente


def generar_oponente_patrones(n_rondas=200):
    """
    Oponente que sigue patrones repetitivos
    Ejemplo: piedra-piedra-papel, piedra-piedra-papel, ...
    """
    np.random.seed(44)

    # Definir patrón base
    patron_base = ['piedra', 'piedra', 'papel', 'tijera', 'papel']

    jugadas_oponente = []
    jugadas_jugador = []

    i = 0
    while len(jugadas_oponente) < n_rondas:
        # Seguir patrón con algo de ruido
        if np.random.random() < 0.85:  # 85% sigue el patrón
            jug_op = patron_base[i % len(patron_base)]
        else:  # 15% aleatorio
            jug_op = np.random.choice(['piedra', 'papel', 'tijera'])

        jugadas_oponente.append(jug_op)

        # Jugador intenta detectar patrón
        if len(jugadas_oponente) >= len(patron_base):
            # Predecir siguiente del patrón
            counter = {'piedra': 'papel', 'papel': 'tijera', 'tijera': 'piedra'}
            if np.random.random() < 0.6:
                siguiente_predicho = patron_base[(i + 1) % len(patron_base)]
                jug_jug = counter[siguiente_predicho]
            else:
                jug_jug = np.random.choice(['piedra', 'papel', 'tijera'])
        else:
            jug_jug = np.random.choice(['piedra', 'papel', 'tijera'])

        jugadas_jugador.append(jug_jug)
        i += 1

    return jugadas_jugador[:n_rondas], jugadas_oponente[:n_rondas]


def generar_dataset_completo(tipo='basico', n_rondas=200, archivo='datos_ppt_ejemplo.csv'):
    """
    Genera un dataset completo y lo guarda en CSV

    Args:
        tipo: 'basico', 'adaptativo', o 'patrones'
        n_rondas: número de rondas
        archivo: nombre del archivo CSV
    """
    print("=" * 70)
    print("GENERADOR DE DATOS DE PPT")
    print("=" * 70)

    # Generar según tipo
    if tipo == 'basico':
        print(f"\nGenerando oponente BÁSICO ({n_rondas} rondas)...")
        print("  Patrón: Prefiere piedra (40%), papel (35%), tijera (25%)")
        jugadas_jugador, jugadas_oponente = generar_oponente_basico(n_rondas)

    elif tipo == 'adaptativo':
        print(f"\nGenerando oponente ADAPTATIVO ({n_rondas} rondas)...")
        print("  Patrón: Si gana repite, si pierde cambia")
        jugadas_jugador, jugadas_oponente = generar_oponente_adaptativo(n_rondas)

    elif tipo == 'patrones':
        print(f"\nGenerando oponente CON PATRONES ({n_rondas} rondas)...")
        print("  Patrón: Secuencia repetitiva")
        jugadas_jugador, jugadas_oponente = generar_oponente_patrones(n_rondas)

    else:
        print(f" Error: tipo '{tipo}' no reconocido")
        return

    # Crear DataFrame
    df = pd.DataFrame({
        'numero_ronda': range(1, n_rondas + 1),
        'jugada_jugador': jugadas_jugador,
        'jugada_oponente': jugadas_oponente
    })

    # Calcular resultados
    def calcular_resultado(jug, op):
        if jug == op:
            return 'empate'
        gana = {'piedra': 'tijera', 'papel': 'piedra', 'tijera': 'papel'}
        return 'victoria' if gana[jug] == op else 'derrota'

    df['resultado'] = df.apply(
        lambda row: calcular_resultado(row['jugada_jugador'], row['jugada_oponente']),
        axis=1
    )

    # Guardar CSV
    df.to_csv(archivo, index=False)
    print(f"\n Archivo guardado: {archivo}")

    # Estadísticas
    print(f"\nEstadísticas del dataset:")
    print(f"  Total rondas: {len(df)}")

    print(f"\n  Jugadas del oponente:")
    for jugada in ['piedra', 'papel', 'tijera']:
        count = (df['jugada_oponente'] == jugada).sum()
        print(f"    {jugada:7s}: {count:3d} ({count/len(df):.1%})")

    print(f"\n  Resultados del jugador:")
    for resultado in ['victoria', 'derrota', 'empate']:
        count = (df['resultado'] == resultado).sum()
        print(f"    {resultado:8s}: {count:3d} ({count/len(df):.1%})")

    # Mostrar primeras filas
    print(f"\nPrimeras 10 rondas:")
    print(df.head(10).to_string(index=False))

    return df


def generar_todos_los_datasets():
    """Genera los 3 tipos de datasets"""
    print("\n" + "" * 35)
    print("GENERANDO TODOS LOS DATASETS")
    print("" * 35 + "\n")

    generar_dataset_completo('basico', 200, 'datos_ppt_basico.csv')
    print("\n")

    generar_dataset_completo('adaptativo', 200, 'datos_ppt_adaptativo.csv')
    print("\n")

    generar_dataset_completo('patrones', 200, 'datos_ppt_patrones.csv')

    print("\n" + "" * 35)
    print("DATASETS GENERADOS")
    print("" * 35)
    print("\nArchivos creados:")
    print("  1. datos_ppt_basico.csv      - Oponente con preferencias fijas")
    print("  2. datos_ppt_adaptativo.csv  - Oponente que adapta estrategia")
    print("  3. datos_ppt_patrones.csv    - Oponente con patrones repetitivos")

    print("\nUso:")
    print("  En los ejercicios, carga estos CSVs con:")
    print("  df = pd.read_csv('datos_ppt_basico.csv')")


if __name__ == "__main__":
    # Generar dataset por defecto (básico)
    generar_dataset_completo('basico', 200, 'datos_ppt_ejemplo.csv')

    # O generar todos
    # generar_todos_los_datasets()
