"""
Ejemplos prácticos de listas en Python.
Las listas son mutables, ordenadas y permiten duplicados.
"""


def ejemplos_basicos():
    """Operaciones básicas con listas."""
    print("=" * 60)
    print("EJEMPLOS BÁSICOS DE LISTAS")
    print("=" * 60)
    
    # Crear listas de diferentes formas
    vacia = []
    numeros = [1, 2, 3, 4, 5]
    mixta = [1, "hola", 3.14, True]  # Python permite tipos mezclados
    
    print(f"Lista vacía: {vacia}")
    print(f"Lista de números: {numeros}")
    print(f"Lista mixta: {mixta}")
    print(f"Tipo de 'numeros': {type(numeros)}")
    print()
    
    # Acceso por índice (como en Java, empezamos en 0)
    print(f"Primer elemento: {numeros[0]}")
    print(f"Tercer elemento: {numeros[2]}")
    print(f"Último elemento: {numeros[-1]}")  # Índice negativo: desde el final
    print(f"Penúltimo elemento: {numeros[-2]}")
    print()


def operaciones_modificacion():
    """Modificar listas: añadir, eliminar, cambiar."""
    print("=" * 60)
    print("MODIFICACIÓN DE LISTAS")
    print("=" * 60)
    
    frutas = ["manzana", "pera"]
    print(f"Lista inicial: {frutas}")
    
    # Añadir elementos
    frutas.append("naranja")  # Al final
    print(f"Después de append('naranja'): {frutas}")
    
    frutas.insert(1, "plátano")  # En posición específica
    print(f"Después de insert(1, 'plátano'): {frutas}")
    
    # Modificar elemento
    frutas[0] = "manzana verde"
    print(f"Después de cambiar frutas[0]: {frutas}")
    
    # Eliminar elementos
    frutas.remove("pera")  # Por valor (elimina la primera ocurrencia)
    print(f"Después de remove('pera'): {frutas}")
    
    ultimo = frutas.pop()  # Elimina y devuelve el último
    print(f"Elemento eliminado con pop(): {ultimo}")
    print(f"Lista después de pop(): {frutas}")
    
    del frutas[0]  # Eliminar por índice
    print(f"Después de del frutas[0]: {frutas}")
    print()


def slicing():
    """Slicing: obtener sublistas."""
    print("=" * 60)
    print("SLICING (REBANADO) DE LISTAS")
    print("=" * 60)
    
    numeros = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"Lista original: {numeros}")
    
    # Sintaxis: lista[inicio:fin:paso]
    # inicio: incluido, fin: excluido
    
    print(f"numeros[2:5]: {numeros[2:5]}")  # índices 2, 3, 4
    print(f"numeros[:4]: {numeros[:4]}")    # desde el inicio hasta índice 3
    print(f"numeros[5:]: {numeros[5:]}")    # desde índice 5 hasta el final
    print(f"numeros[::2]: {numeros[::2]}")  # todos, de 2 en 2
    print(f"numeros[1::2]: {numeros[1::2]}")  # impares
    print(f"numeros[::-1]: {numeros[::-1]}")  # invertir la lista
    print()


def metodos_utiles():
    """Métodos útiles de listas."""
    print("=" * 60)
    print("MÉTODOS ÚTILES")
    print("=" * 60)
    
    numeros = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Lista original: {numeros}")
    
    # Longitud
    print(f"Longitud: {len(numeros)}")
    
    # Contar ocurrencias
    print(f"Veces que aparece el 1: {numeros.count(1)}")
    
    # Buscar índice de un elemento
    print(f"Índice del 4: {numeros.index(4)}")
    
    # Ordenar
    numeros_copia = numeros.copy()  # Hacer una copia
    numeros_copia.sort()  # Ordena en el sitio (modifica la lista)
    print(f"Lista ordenada (sort): {numeros_copia}")
    print(f"Lista original sin cambios: {numeros}")
    
    # sorted() devuelve una nueva lista ordenada
    numeros_ordenados = sorted(numeros)
    print(f"sorted(numeros): {numeros_ordenados}")
    
    # Invertir
    numeros_copia.reverse()
    print(f"Después de reverse(): {numeros_copia}")
    
    # Sumar, máximo, mínimo
    print(f"Suma: {sum(numeros)}")
    print(f"Máximo: {max(numeros)}")
    print(f"Mínimo: {min(numeros)}")
    print()


def comprension_listas():
    """List comprehensions: forma compacta de crear listas."""
    print("=" * 60)
    print("COMPRENSIÓN DE LISTAS (List Comprehensions)")
    print("=" * 60)
    
    # Forma tradicional: crear lista de cuadrados
    cuadrados_tradicional = []
    for i in range(1, 6):
        cuadrados_tradicional.append(i ** 2)
    print(f"Cuadrados (forma tradicional): {cuadrados_tradicional}")
    
    # Forma pythónica: list comprehension
    cuadrados = [i ** 2 for i in range(1, 6)]
    print(f"Cuadrados (comprehension): {cuadrados}")
    
    # Con condición: solo números pares
    pares = [i for i in range(1, 11) if i % 2 == 0]
    print(f"Números pares del 1 al 10: {pares}")
    
    # Más complejo: transformar strings
    nombres = ["ana", "CARLOS", "BeatRiz"]
    nombres_normalizados = [n.capitalize() for n in nombres]
    print(f"Nombres normalizados: {nombres_normalizados}")
    print()


def listas_anidadas():
    """Listas dentro de listas (matrices)."""
    print("=" * 60)
    print("LISTAS ANIDADAS")
    print("=" * 60)
    
    # Matriz 3x3
    matriz = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Matriz 3x3:")
    for fila in matriz:
        print(fila)
    
    print(f"\nElemento en fila 1, columna 2: {matriz[1][2]}")  # 6
    print(f"Primera fila: {matriz[0]}")
    print(f"Tercera columna: {[fila[2] for fila in matriz]}")
    print()


def errores_comunes():
    """Errores típicos al trabajar con listas."""
    print("=" * 60)
    print("ERRORES COMUNES")
    print("=" * 60)
    
    lista = [1, 2, 3]
    
    # Error 1: índice fuera de rango
    try:
        print(lista[10])
    except IndexError as e:
        print(f"IndexError: {e}")
    
    # Error 2: intentar eliminar algo que no existe
    try:
        lista.remove(99)
    except ValueError as e:
        print(f"ValueError: {e}")
    
    # Error 3: modificar lista mientras se itera (¡cuidado!)
    print("\n¡Cuidado al modificar mientras iteras!")
    numeros = [1, 2, 3, 4, 5]
    print(f"Lista original: {numeros}")
    
    # MAL: esto puede dar resultados inesperados
    # for num in numeros:
    #     if num % 2 == 0:
    #         numeros.remove(num)  # ¡No hacer así!
    
    # BIEN: iterar sobre una copia
    numeros_copia = numeros.copy()
    for num in numeros_copia:
        if num % 2 == 0:
            numeros.remove(num)
    print(f"Después de eliminar pares (correcto): {numeros}")
    
    # MEJOR: usar comprensión de listas
    numeros = [1, 2, 3, 4, 5]
    impares = [n for n in numeros if n % 2 != 0]
    print(f"Solo impares (con comprehension): {impares}")
    print()


if __name__ == "__main__":
    ejemplos_basicos()
    operaciones_modificacion()
    slicing()
    metodos_utiles()
    comprension_listas()
    listas_anidadas()
    errores_comunes()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE LISTAS")
    print("=" * 60)
