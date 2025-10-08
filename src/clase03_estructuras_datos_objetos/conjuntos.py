"""
Ejemplos prácticos de conjuntos (sets) en Python.
Los conjuntos son colecciones sin orden y sin duplicados, mutables.
"""


def ejemplos_basicos():
    """Operaciones básicas con conjuntos."""
    print("=" * 60)
    print("EJEMPLOS BÁSICOS DE CONJUNTOS")
    print("=" * 60)
    
    # Crear conjuntos
    vacio = set()  # OJO: {} crea un diccionario vacío, no un conjunto
    numeros = {1, 2, 3, 4, 5}
    letras = set("hola")  # Crea conjunto desde un iterable
    
    print(f"Conjunto vacío: {vacio}")
    print(f"Números: {numeros}")
    print(f"Letras de 'hola': {letras}")  # Nota: sin orden y sin duplicados
    print(f"Tipo: {type(numeros)}")
    print()
    
    # Los conjuntos eliminan duplicados automáticamente
    con_duplicados = {1, 2, 2, 3, 3, 3, 4}
    print(f"Con duplicados {1, 2, 2, 3, 3, 3, 4}: {con_duplicados}")
    print()
    
    # Crear conjunto desde lista (elimina duplicados)
    lista = [1, 2, 2, 3, 3, 3, 4, 4]
    conjunto = set(lista)
    print(f"Lista: {lista}")
    print(f"Conjunto desde lista: {conjunto}")
    print()


def operaciones_basicas():
    """Añadir, eliminar elementos."""
    print("=" * 60)
    print("OPERACIONES BÁSICAS")
    print("=" * 60)
    
    frutas = {"manzana", "pera"}
    print(f"Conjunto inicial: {frutas}")
    
    # Añadir elemento
    frutas.add("naranja")
    print(f"Después de add('naranja'): {frutas}")
    
    # Añadir elemento duplicado (no pasa nada)
    frutas.add("manzana")
    print(f"Después de add('manzana') de nuevo: {frutas}")
    
    # Eliminar con remove (error si no existe)
    frutas.remove("pera")
    print(f"Después de remove('pera'): {frutas}")
    
    try:
        frutas.remove("plátano")  # Error: no existe
    except KeyError as e:
        print(f"Error al remove('plátano'): {e}")
    
    # Eliminar con discard (sin error si no existe)
    frutas.discard("plátano")  # No pasa nada
    print(f"Después de discard('plátano'): {frutas}")
    
    # pop: elimina y devuelve un elemento arbitrario
    elemento = frutas.pop()
    print(f"pop() devolvió: {elemento}")
    print(f"Conjunto después de pop: {frutas}")
    
    # clear: vaciar conjunto
    frutas.clear()
    print(f"Después de clear(): {frutas}")
    print()


def pertenencia():
    """Comprobar si un elemento está en el conjunto (muy rápido)."""
    print("=" * 60)
    print("COMPROBAR PERTENENCIA (MUY RÁPIDO)")
    print("=" * 60)
    
    numeros_pares = {0, 2, 4, 6, 8, 10}
    
    # in: pertenencia
    print(f"Conjunto: {numeros_pares}")
    print(f"¿4 está en el conjunto? {4 in numeros_pares}")
    print(f"¿5 está en el conjunto? {5 in numeros_pares}")
    print()
    
    # Ventaja: la búsqueda en conjuntos es O(1), en listas es O(n)
    # Para colecciones grandes, los conjuntos son mucho más rápidos
    print("Nota: buscar en conjuntos es mucho más rápido que en listas.")
    print()


def operaciones_conjuntos():
    """Operaciones matemáticas de conjuntos."""
    print("=" * 60)
    print("OPERACIONES MATEMÁTICAS DE CONJUNTOS")
    print("=" * 60)
    
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7, 8}
    
    print(f"Conjunto A: {a}")
    print(f"Conjunto B: {b}")
    print()
    
    # Unión: elementos en A o en B (o en ambos)
    union1 = a.union(b)
    union2 = a | b  # Operador
    print(f"Unión A ∪ B (método): {union1}")
    print(f"Unión A | B (operador): {union2}")
    print()
    
    # Intersección: elementos en A y en B
    interseccion1 = a.intersection(b)
    interseccion2 = a & b  # Operador
    print(f"Intersección A ∩ B (método): {interseccion1}")
    print(f"Intersección A & B (operador): {interseccion2}")
    print()
    
    # Diferencia: elementos en A pero no en B
    diferencia1 = a.difference(b)
    diferencia2 = a - b  # Operador
    print(f"Diferencia A - B (método): {diferencia1}")
    print(f"Diferencia A - B (operador): {diferencia2}")
    print()
    
    # Diferencia simétrica: elementos en A o en B, pero no en ambos
    dif_simetrica1 = a.symmetric_difference(b)
    dif_simetrica2 = a ^ b  # Operador
    print(f"Diferencia simétrica A △ B (método): {dif_simetrica1}")
    print(f"Diferencia simétrica A ^ B (operador): {dif_simetrica2}")
    print()


def relaciones_conjuntos():
    """Subconjuntos, superconjuntos, disjuntos."""
    print("=" * 60)
    print("RELACIONES ENTRE CONJUNTOS")
    print("=" * 60)
    
    a = {1, 2, 3}
    b = {1, 2, 3, 4, 5}
    c = {6, 7, 8}
    
    print(f"A: {a}")
    print(f"B: {b}")
    print(f"C: {c}")
    print()
    
    # Subconjunto: todos los elementos de A están en B
    print(f"¿A es subconjunto de B? {a.issubset(b)} (A ⊆ B)")
    print(f"¿B es subconjunto de A? {b.issubset(a)}")
    print()
    
    # Superconjunto: todos los elementos de B están en A
    print(f"¿B es superconjunto de A? {b.issuperset(a)} (B ⊇ A)")
    print(f"¿A es superconjunto de B? {a.issuperset(b)}")
    print()
    
    # Disjuntos: no tienen elementos en común
    print(f"¿A y B son disjuntos? {a.isdisjoint(b)}")
    print(f"¿A y C son disjuntos? {a.isdisjoint(c)}")
    print()


def casos_uso_practicos():
    """Casos de uso prácticos de conjuntos."""
    print("=" * 60)
    print("CASOS DE USO PRÁCTICOS")
    print("=" * 60)
    
    # 1. Eliminar duplicados de una lista
    print("1. Eliminar duplicados de una lista:")
    numeros = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    sin_duplicados = list(set(numeros))
    print(f"   Lista original: {numeros}")
    print(f"   Sin duplicados: {sin_duplicados}")
    print()
    
    # 2. Encontrar elementos únicos en dos listas
    print("2. Encontrar letras únicas en dos palabras:")
    palabra1 = "python"
    palabra2 = "java"
    letras1 = set(palabra1)
    letras2 = set(palabra2)
    comunes = letras1 & letras2
    solo_python = letras1 - letras2
    solo_java = letras2 - letras1
    print(f"   Palabra 1: '{palabra1}' -> letras: {letras1}")
    print(f"   Palabra 2: '{palabra2}' -> letras: {letras2}")
    print(f"   Letras comunes: {comunes}")
    print(f"   Solo en Python: {solo_python}")
    print(f"   Solo en Java: {solo_java}")
    print()
    
    # 3. Verificar pertenencia rápida
    print("3. Lista de IPs permitidas (búsqueda rápida):")
    ips_permitidas = {"192.168.1.1", "192.168.1.2", "10.0.0.1"}
    ip_cliente = "192.168.1.1"
    if ip_cliente in ips_permitidas:
        print(f"   IP {ip_cliente}: ACCESO PERMITIDO")
    else:
        print(f"   IP {ip_cliente}: ACCESO DENEGADO")
    print()
    
    # 4. Seguimiento de elementos vistos
    print("4. Detectar duplicados mientras procesamos:")
    ids_vistos = set()
    ids_entrantes = [101, 102, 103, 102, 104, 101, 105]
    print(f"   IDs entrantes: {ids_entrantes}")
    for id in ids_entrantes:
        if id in ids_vistos:
            print(f"   ¡Duplicado detectado! ID: {id}")
        else:
            ids_vistos.add(id)
            print(f"   Nuevo ID registrado: {id}")
    print()


def ejercicio_asignaturas():
    """Ejercicio: análisis de asignaturas de estudiantes."""
    print("=" * 60)
    print("EJERCICIO: ASIGNATURAS DE ESTUDIANTES")
    print("=" * 60)
    
    # Asignaturas de cada estudiante
    ana = {"Matemáticas", "Física", "Programación", "Inglés"}
    carlos = {"Física", "Química", "Programación", "Historia"}
    beatriz = {"Matemáticas", "Programación", "Química", "Arte"}
    
    print(f"Ana: {ana}")
    print(f"Carlos: {carlos}")
    print(f"Beatriz: {beatriz}")
    print()
    
    # ¿Qué asignaturas cursan todos?
    todas = ana & carlos & beatriz
    print(f"Asignaturas que cursan todos: {todas}")
    
    # ¿Qué asignaturas cursa al menos uno?
    alguno = ana | carlos | beatriz
    print(f"Asignaturas que cursa al menos uno: {alguno}")
    
    # ¿Qué asignaturas cursa Ana pero no Carlos?
    solo_ana = ana - carlos
    print(f"Asignaturas solo de Ana (no Carlos): {solo_ana}")
    
    # ¿Qué asignaturas son exclusivas de cada uno?
    exclusivas_ana = ana - carlos - beatriz
    exclusivas_carlos = carlos - ana - beatriz
    exclusivas_beatriz = beatriz - ana - carlos
    print(f"Exclusivas de Ana: {exclusivas_ana}")
    print(f"Exclusivas de Carlos: {exclusivas_carlos}")
    print(f"Exclusivas de Beatriz: {exclusivas_beatriz}")
    print()


def conjuntos_inmutables():
    """frozenset: conjuntos inmutables."""
    print("=" * 60)
    print("FROZENSET: CONJUNTOS INMUTABLES")
    print("=" * 60)
    
    # frozenset: como set pero inmutable
    normal = {1, 2, 3}
    inmutable = frozenset([1, 2, 3])
    
    print(f"Set normal: {normal}")
    print(f"Frozenset: {inmutable}")
    print()
    
    # Se puede modificar el set normal
    normal.add(4)
    print(f"Después de add(4): {normal}")
    
    # No se puede modificar el frozenset
    try:
        inmutable.add(4)
    except AttributeError as e:
        print(f"Error al intentar modificar frozenset: {e}")
    print()
    
    # Ventaja: frozenset se puede usar como clave de diccionario
    print("Uso de frozenset como clave de diccionario:")
    mapa = {
        frozenset([1, 2]): "par (1,2)",
        frozenset([3, 4]): "par (3,4)"
    }
    print(f"Diccionario con frozensets como claves: {mapa}")
    print()


def comprension_conjuntos():
    """Comprensión de conjuntos."""
    print("=" * 60)
    print("COMPRENSIÓN DE CONJUNTOS")
    print("=" * 60)
    
    # Similar a list comprehension pero con {}
    cuadrados = {x**2 for x in range(1, 6)}
    print(f"Cuadrados de 1 a 5: {cuadrados}")
    
    # Con condición
    pares = {x for x in range(1, 11) if x % 2 == 0}
    print(f"Números pares del 1 al 10: {pares}")
    
    # Eliminar duplicados de forma elegante
    texto = "hello world"
    letras_unicas = {c for c in texto if c.isalpha()}
    print(f"Letras únicas en '{texto}': {letras_unicas}")
    print()


if __name__ == "__main__":
    ejemplos_basicos()
    operaciones_basicas()
    pertenencia()
    operaciones_conjuntos()
    relaciones_conjuntos()
    casos_uso_practicos()
    ejercicio_asignaturas()
    conjuntos_inmutables()
    comprension_conjuntos()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE CONJUNTOS")
    print("=" * 60)
