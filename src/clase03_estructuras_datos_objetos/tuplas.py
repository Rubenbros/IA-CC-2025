"""
Ejemplos prácticos de tuplas en Python.
Las tuplas son inmutables, ordenadas y permiten duplicados.
"""


def ejemplos_basicos():
    """Operaciones básicas con tuplas."""
    print("=" * 60)
    print("EJEMPLOS BÁSICOS DE TUPLAS")
    print("=" * 60)
    
    # Crear tuplas
    vacia = ()
    una_elemento = (42,)  # ¡Coma obligatoria para tupla de un elemento!
    coordenadas = (10, 20)
    punto_3d = (5, 10, 15)
    mixta = (1, "hola", 3.14, True)
    sin_parentesis = 1, 2, 3  # También es una tupla
    
    print(f"Tupla vacía: {vacia}")
    print(f"Un elemento: {una_elemento}")
    print(f"Coordenadas 2D: {coordenadas}")
    print(f"Punto 3D: {punto_3d}")
    print(f"Tupla mixta: {mixta}")
    print(f"Sin paréntesis: {sin_parentesis}")
    print(f"Tipo: {type(coordenadas)}")
    print()
    
    # Acceso (igual que listas)
    print(f"Primera coordenada: {punto_3d[0]}")
    print(f"Última coordenada: {punto_3d[-1]}")
    print()


def inmutabilidad():
    """Las tuplas NO se pueden modificar."""
    print("=" * 60)
    print("INMUTABILIDAD DE LAS TUPLAS")
    print("=" * 60)
    
    coordenadas = (10, 20)
    print(f"Tupla original: {coordenadas}")
    
    # Esto funciona (acceso)
    print(f"x = {coordenadas[0]}, y = {coordenadas[1]}")
    
    # Esto NO funciona (modificación)
    try:
        coordenadas[0] = 30
    except TypeError as e:
        print(f"Error al intentar modificar: {e}")
    
    # Si necesitas modificar, convierte a lista, modifica y vuelve a tupla
    lista_temp = list(coordenadas)
    lista_temp[0] = 30
    coordenadas_nuevas = tuple(lista_temp)
    print(f"Nueva tupla (convertida desde lista): {coordenadas_nuevas}")
    print()


def desempaquetado():
    """Desempaquetado de tuplas: asignar a múltiples variables."""
    print("=" * 60)
    print("DESEMPAQUETADO DE TUPLAS")
    print("=" * 60)
    
    # Desempaquetado básico
    punto = (10, 20)
    x, y = punto  # Desempaquetar en dos variables
    print(f"Punto: {punto}")
    print(f"x = {x}, y = {y}")
    print()
    
    # Con más elementos
    fecha = (2025, 1, 15)
    año, mes, dia = fecha
    print(f"Fecha: {fecha}")
    print(f"Año: {año}, Mes: {mes}, Día: {dia}")
    print()
    
    # Intercambiar variables (¡muy pythónico!)
    a, b = 5, 10
    print(f"Antes: a={a}, b={b}")
    a, b = b, a  # Intercambio en una línea
    print(f"Después: a={a}, b={b}")
    print()
    
    # Desempaquetado con *resto
    numeros = (1, 2, 3, 4, 5)
    primero, segundo, *resto = numeros
    print(f"Tupla: {numeros}")
    print(f"Primero: {primero}, Segundo: {segundo}, Resto: {resto}")
    print()


def retorno_multiple():
    """Funciones que devuelven múltiples valores (realmente una tupla)."""
    print("=" * 60)
    print("RETORNO MÚLTIPLE DE FUNCIONES")
    print("=" * 60)
    
    def dividir(a, b):
        """Devuelve cociente y resto."""
        cociente = a // b
        resto = a % b
        return cociente, resto  # Realmente devuelve una tupla
    
    # Capturar ambos valores
    c, r = dividir(17, 5)
    print(f"17 dividido entre 5: cociente={c}, resto={r}")
    
    # O capturar como tupla
    resultado = dividir(17, 5)
    print(f"Resultado completo: {resultado} (tipo: {type(resultado)})")
    print()
    
    def obtener_stats(numeros):
        """Devuelve min, max y promedio."""
        return min(numeros), max(numeros), sum(numeros) / len(numeros)
    
    datos = [10, 20, 30, 40, 50]
    minimo, maximo, promedio = obtener_stats(datos)
    print(f"Stats de {datos}:")
    print(f"  Mínimo: {minimo}, Máximo: {maximo}, Promedio: {promedio}")
    print()


def tuplas_vs_listas():
    """Cuándo usar tuplas vs listas."""
    print("=" * 60)
    print("TUPLAS VS LISTAS: ¿CUÁNDO USAR CADA UNA?")
    print("=" * 60)
    
    # Usar TUPLAS para:
    # 1. Datos que no deben cambiar
    configuracion = ("localhost", 8080, "admin")
    print(f"Configuración (no debe cambiar): {configuracion}")
    
    # 2. Coordenadas, puntos
    posicion = (45.5, -73.6)  # Latitud, longitud
    print(f"Posición GPS: {posicion}")
    
    # 3. Claves de diccionario (las tuplas son hashables, las listas no)
    mapa = {
        (0, 0): "origen",
        (1, 0): "derecha",
        (0, 1): "arriba"
    }
    print(f"Mapa con tuplas como claves: {mapa}")
    
    # 4. Retornar múltiples valores
    def obtener_nombre_completo():
        return "Ana", "García", "López"
    
    nombre, apellido1, apellido2 = obtener_nombre_completo()
    print(f"Nombre completo: {nombre} {apellido1} {apellido2}")
    print()
    
    # Usar LISTAS para:
    # 1. Colecciones que cambiarán
    tareas = ["estudiar", "programar"]
    tareas.append("descansar")
    print(f"Tareas (pueden cambiar): {tareas}")
    
    # 2. Cuando necesites métodos como sort, append, etc.
    numeros = [3, 1, 4, 1, 5]
    numeros.sort()
    print(f"Números ordenados: {numeros}")
    print()


def metodos_tuplas():
    """Métodos disponibles en tuplas (son pocos)."""
    print("=" * 60)
    print("MÉTODOS DE TUPLAS")
    print("=" * 60)
    
    numeros = (1, 2, 3, 2, 4, 2, 5)
    print(f"Tupla: {numeros}")
    
    # count: contar ocurrencias
    print(f"Veces que aparece el 2: {numeros.count(2)}")
    
    # index: buscar posición de primera ocurrencia
    print(f"Índice del primer 3: {numeros.index(3)}")
    
    # len (función, no método)
    print(f"Longitud: {len(numeros)}")
    
    # También funcionan: sum, min, max
    print(f"Suma: {sum(numeros)}")
    print(f"Mínimo: {min(numeros)}, Máximo: {max(numeros)}")
    print()


def tuplas_anidadas():
    """Tuplas dentro de tuplas."""
    print("=" * 60)
    print("TUPLAS ANIDADAS")
    print("=" * 60)
    
    # Representar un rectángulo con dos puntos
    rectangulo = ((0, 0), (10, 5))  # esquina inferior izq, superior der
    print(f"Rectángulo: {rectangulo}")
    print(f"Esquina inferior izquierda: {rectangulo[0]}")
    print(f"Esquina superior derecha: {rectangulo[1]}")
    
    # Desempaquetar anidado
    (x1, y1), (x2, y2) = rectangulo
    print(f"Coordenadas: ({x1}, {y1}) a ({x2}, {y2})")
    print()
    
    # Lista de tuplas (muy común)
    estudiantes = [
        ("Ana", 20, 8.5),
        ("Carlos", 22, 7.0),
        ("Beatriz", 21, 9.0)
    ]
    print("Estudiantes (lista de tuplas):")
    for nombre, edad, nota in estudiantes:
        print(f"  {nombre}, {edad} años, nota: {nota}")
    print()


def conversiones():
    """Convertir entre tuplas, listas y otros tipos."""
    print("=" * 60)
    print("CONVERSIONES")
    print("=" * 60)
    
    # Lista a tupla
    lista = [1, 2, 3]
    tupla = tuple(lista)
    print(f"Lista: {lista} -> Tupla: {tupla}")
    
    # Tupla a lista
    tupla = (4, 5, 6)
    lista = list(tupla)
    print(f"Tupla: {tupla} -> Lista: {lista}")
    
    # String a tupla
    texto = "hola"
    tupla = tuple(texto)
    print(f"String: '{texto}' -> Tupla: {tupla}")
    
    # Range a tupla
    tupla = tuple(range(5))
    print(f"range(5) -> Tupla: {tupla}")
    print()


if __name__ == "__main__":
    ejemplos_basicos()
    inmutabilidad()
    desempaquetado()
    retorno_multiple()
    tuplas_vs_listas()
    metodos_tuplas()
    tuplas_anidadas()
    conversiones()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE TUPLAS")
    print("=" * 60)
