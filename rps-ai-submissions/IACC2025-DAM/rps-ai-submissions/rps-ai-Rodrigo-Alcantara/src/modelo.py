import random
import csv
import os
from collections import defaultdict

# Configuración del juego
MOVIMIENTOS = ['piedra', 'papel', 'tijera']

# Diccionario que define quien gana dependiendo de lo que saque cada uno
TABLA_GANADORES = {
    ('piedra', 'tijera'): 'jugador',
    ('papel', 'piedra'): 'jugador',
    ('tijera', 'papel'): 'jugador',
    ('tijera', 'piedra'): 'ia',
    ('piedra', 'papel'): 'ia',
    ('papel', 'tijera'): 'ia'
}

CONTRAATAQUE = {
    'piedra': 'papel',
    'papel': 'tijera',
    'tijera': 'piedra'
}


class AnalizadorPatrones:
    """Analiza y aprende de los movimientos del jugador"""

    def __init__(self):
        self.jugadas_anteriores = []
        self.cadenas_markov = defaultdict(lambda: defaultdict(int))
        self.secuencias_dobles = defaultdict(lambda: defaultdict(int))
        self.ventana_analisis = []
        self.max_ventana = 10

        # Sistema de puntuación para cada métod
        self.puntos_metodos = {
            'cadena_simple': 1.0,
            'cadena_doble': 1.5,
            'mas_frecuente': 1.0,
            'menos_frecuente': 1.2,
            'ciclos': 1.3,
            'ventana': 1.0
        }

        self.aciertos_metodos = defaultdict(int)

    def registrar_jugada(self, movimiento):
        """Almacena el movimiento y actualiza las estructuras de datos"""
        if len(self.jugadas_anteriores) > 0:
            anterior = self.jugadas_anteriores[-1]
            self.cadenas_markov[anterior][movimiento] += 1

        if len(self.jugadas_anteriores) >= 2:
            par_anterior = (self.jugadas_anteriores[-2], self.jugadas_anteriores[-1])
            self.secuencias_dobles[par_anterior][movimiento] += 1

        self.jugadas_anteriores.append(movimiento)
        self.ventana_analisis.append(movimiento)

        if len(self.ventana_analisis) > self.max_ventana:
            self.ventana_analisis.pop(0)

    def metodo_cadena_simple(self):
        """Predice basándose en el último movimiento"""
        if not self.jugadas_anteriores:
            return None

        ultimo = self.jugadas_anteriores[-1]
        opciones = self.cadenas_markov[ultimo]

        if not opciones:
            return None

        return max(opciones, key=opciones.get)

    def metodo_cadena_doble(self):
        """Predice usando los dos últimos movimientos"""
        if len(self.jugadas_anteriores) < 2:
            return None

        par = (self.jugadas_anteriores[-2], self.jugadas_anteriores[-1])
        opciones = self.secuencias_dobles[par]

        if not opciones:
            return None

        return max(opciones, key=opciones.get)

    def metodo_frecuencia_maxima(self):
        """Encuentra el movimiento más usado"""
        if len(self.jugadas_anteriores) < 3:
            return None

        conteo = defaultdict(int)
        for mov in self.jugadas_anteriores:
            conteo[mov] += 1

        return max(conteo, key=conteo.get)

    def metodo_frecuencia_minima(self):
        """Encuentra el movimiento menos usado (para jugadores adaptativos)"""
        if len(self.jugadas_anteriores) < 5:
            return None

        conteo = defaultdict(int)
        for mov in self.jugadas_anteriores:
            conteo[mov] += 1

        return min(conteo, key=conteo.get)

    def metodo_deteccion_ciclos(self):
        """Busca patrones que se repiten"""
        if len(self.jugadas_anteriores) < 6:
            return None

        for longitud in range(2, min(5, len(self.jugadas_anteriores) // 2 + 1)):
            patron_actual = self.jugadas_anteriores[-longitud:]
            patron_previo = self.jugadas_anteriores[-2 * longitud:-longitud]

            if patron_actual == patron_previo:
                if len(self.jugadas_anteriores) > 2 * longitud:
                    return self.jugadas_anteriores[-longitud]

        return None

    def metodo_ventana_reciente(self):
        """Analiza solo las últimas jugadas para detectar tendencias"""
        if len(self.ventana_analisis) < 5:
            return None

        conteo = defaultdict(int)
        for mov in self.ventana_analisis[-5:]:
            conteo[mov] += 1

        mov_dominante = max(conteo, key=conteo.get)
        if conteo[mov_dominante] >= 3:
            return mov_dominante

        return None

    def obtener_prediccion(self):
        """Combina todos los métodos usando votación ponderada"""
        predicciones = {}

        metodos = [
            ('cadena_simple', self.metodo_cadena_simple()),
            ('cadena_doble', self.metodo_cadena_doble()),
            ('mas_frecuente', self.metodo_frecuencia_maxima()),
            ('menos_frecuente', self.metodo_frecuencia_minima()),
            ('ciclos', self.metodo_deteccion_ciclos()),
            ('ventana', self.metodo_ventana_reciente())
        ]

        for nombre_metodo, resultado in metodos:
            if resultado:
                predicciones[nombre_metodo] = resultado

        if not predicciones:
            if len(self.jugadas_anteriores) >= 3:
                return self.metodo_frecuencia_maxima()
            return random.choice(MOVIMIENTOS)

        # Sistema de votación con pesos
        votos = defaultdict(float)
        for metodo, prediccion in predicciones.items():
            peso = self.puntos_metodos[metodo]
            votos[prediccion] += peso

        return max(votos, key=votos.get)

    def actualizar_eficiencia(self, metodo, acerto):
        """Ajusta los pesos según el rendimiento"""
        if acerto:
            self.aciertos_metodos[metodo] += 1
            self.puntos_metodos[metodo] = min(2.5, self.puntos_metodos[metodo] * 1.1)
        else:
            self.puntos_metodos[metodo] = max(0.3, self.puntos_metodos[metodo] * 0.95)


def calcular_resultado(mov_jugador, mov_ia):
    """Determina quién ganó la ronda"""
    if mov_jugador == mov_ia:
        return 'empate'

    return TABLA_GANADORES.get((mov_jugador, mov_ia), 'ia')


def obtener_porcentaje(diccionario, clave, total):
    """Calcula porcentaje de uso de una jugada"""
    cantidad = diccionario.get(clave, 0)
    return round((cantidad / total) * 100, 2) if total > 0 else 0.0


def exportar_a_csv(datos_partida):
    """Guarda la información de la partida en CSV"""
    if not datos_partida:
        print("\nNo hay información para exportar.")
        return

    try:
        ruta_datos = r"C:\Users\rodri\PycharmProjects\rps-ai-Rodrigo-Alcantara\data"
        os.makedirs(ruta_datos, exist_ok=True)

        archivo_salida = os.path.join(ruta_datos, "resultado_partidas.csv")

        encabezados = [
            'numero_ronda', 'jugador', 'IA', 'resultado',
            'racha_victorias_jugador', 'racha_derrotas_jugador',
            'racha_victorias_IA', 'racha_derrotas_IA',
            'pct_piedra_jugador', 'pct_papel_jugador', 'pct_tijera_jugador',
            'pct_piedra_IA', 'pct_papel_IA', 'pct_tijera_IA'
        ]

        with open(archivo_salida, mode='w', newline='', encoding='utf-8') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=encabezados)
            escritor.writeheader()
            escritor.writerows(datos_partida)

        print(f"\n✓ Datos guardados exitosamente en:\n{archivo_salida}")

    except IOError as error:
        print(f"Error al exportar: {error}")


def ejecutar_juego():
    """Función principal del juego"""
    print("=" * 60)
    print("       PIEDRA, PAPEL O TIJERA - IA AVANZADA")
    print("=" * 60)
    print("Comandos: 'piedra' (1), 'papel' (2), 'tijera' (3) o 'salir'\n")

    analizador = AnalizadorPatrones()
    registro_partida = []
    historial_ia = []

    marcador = {'jugador': 0, 'ia': 0, 'empate': 0}
    numero_ronda = 1
    max_rondas = 50

    # Control de rachas
    racha_jugador = 0
    racha_ia = 0
    ganador_anterior = None

    while numero_ronda <= max_rondas:
        print("-" * 60)
        entrada = input(f" Ronda {numero_ronda} >> Elige tu jugada: ").lower().strip()
        # Convertir números a palabras
        if entrada == '1':
            entrada = 'piedra'
        elif entrada == '2':
            entrada = 'papel'
        elif entrada == '3':
            entrada = 'tijera'

        if entrada == 'salir':
            break

        if entrada not in MOVIMIENTOS:
            print("  Error: Jugada inválida. Usa 'piedra', 'papel' o 'tijera'.")
            continue

        # IA realiza predicción
        prediccion = analizador.obtener_prediccion()
        jugada_ia = CONTRAATAQUE[prediccion]

        # Evaluar resultado
        resultado = calcular_resultado(entrada, jugada_ia)

        etiqueta_resultado = "Empate"
        if resultado == 'jugador':
            etiqueta_resultado = "Victoria"
            marcador['jugador'] += 1
        elif resultado == 'ia':
            etiqueta_resultado = "Derrota"
            marcador['ia'] += 1
        else:
            marcador['empate'] += 1

        print(f"   Tu jugada: {entrada.upper()} | IA: {jugada_ia.upper()} => {etiqueta_resultado.upper()}")

        # Gestión de rachas
        if resultado == 'empate':
            racha_jugador = 0
            racha_ia = 0
            ganador_anterior = 'empate'
        elif resultado == 'jugador':
            racha_jugador = racha_jugador + 1 if ganador_anterior == 'jugador' else 1
            racha_ia = 0
            ganador_anterior = 'jugador'
        else:
            racha_ia = racha_ia + 1 if ganador_anterior == 'ia' else 1
            racha_jugador = 0
            ganador_anterior = 'ia'

        derrotas_jugador = racha_ia if ganador_anterior == 'ia' else 0
        derrotas_ia = racha_jugador if ganador_anterior == 'jugador' else 0

        # Calcular y mostrar eficiencia
        partidas_validas = marcador['jugador'] + marcador['ia']
        if partidas_validas > 0:
            tasa_victoria_ia = (marcador['ia'] / partidas_validas) * 100
            print(f"    📊 Eficiencia IA: {tasa_victoria_ia:.2f}% "
                  f"({marcador['ia']}V - {marcador['jugador']}D - {marcador['empate']}E)")
        else:
            print(f"    📊 Eficiencia IA: 0.00% (Sin partidas decisivas)")

        # Actualizar aprendizaje
        analizador.registrar_jugada(entrada)
        historial_ia.append(jugada_ia)

        # Calcular estadísticas
        total_rondas = len(analizador.jugadas_anteriores)
        conteo_jugador = defaultdict(int)
        conteo_ia = defaultdict(int)

        for mov in analizador.jugadas_anteriores:
            conteo_jugador[mov] += 1
        for mov in historial_ia:
            conteo_ia[mov] += 1

        # Preparar datos para CSV
        fila_datos = {
            'numero_ronda': numero_ronda,
            'jugador': entrada,
            'IA': jugada_ia,
            'resultado': etiqueta_resultado,
            'racha_victorias_jugador': racha_jugador,
            'racha_derrotas_jugador': derrotas_jugador,
            'racha_victorias_IA': racha_ia,
            'racha_derrotas_IA': derrotas_ia,
            'pct_piedra_jugador': obtener_porcentaje(conteo_jugador, 'piedra', total_rondas),
            'pct_papel_jugador': obtener_porcentaje(conteo_jugador, 'papel', total_rondas),
            'pct_tijera_jugador': obtener_porcentaje(conteo_jugador, 'tijera', total_rondas),
            'pct_piedra_IA': obtener_porcentaje(conteo_ia, 'piedra', total_rondas),
            'pct_papel_IA': obtener_porcentaje(conteo_ia, 'papel', total_rondas),
            'pct_tijera_IA': obtener_porcentaje(conteo_ia, 'tijera', total_rondas),
        }

        registro_partida.append(fila_datos)
        numero_ronda += 1

    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("                   ESTADÍSTICAS FINALES")
    print("=" * 60)
    print(f"  Rondas jugadas: {numero_ronda - 1}")
    print(f"  Victorias del Jugador: {marcador['jugador']}")
    print(f"  Victorias de la IA: {marcador['ia']}")
    print(f"  Empates: {marcador['empate']}")

    if partidas_validas > 0:
        eficiencia_total = (marcador['ia'] / partidas_validas) * 100
        print(f"\n EFICIENCIA FINAL DE LA IA: {eficiencia_total:.2f}%")

        if eficiencia_total >= 60:
            print("¡Excelente! La IA superó el 60% de efectividad")
        elif eficiencia_total >= 50:
            print("Buen resultado, más del 50% de efectividad")
        else:
            print("La IA necesita más entrenamiento")

    print("=" * 60)

    # Guardado de Datos de la Partida
    if registro_partida:
        print("\n Exportando resultados...")
        exportar_a_csv(registro_partida)
    else:
        print("No se registraron datos para exportar.")


if __name__ == "__main__":
    ejecutar_juego()
