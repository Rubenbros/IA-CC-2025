"""
Ejemplos prácticos de diccionarios en Python.
Los diccionarios son colecciones de pares clave-valor, mutables.
"""


def ejemplos_basicos():
    """Operaciones básicas con diccionarios."""
    print("=" * 60)
    print("EJEMPLOS BÁSICOS DE DICCIONARIOS")
    print("=" * 60)
    
    # Crear diccionarios
    vacio = {}
    vacio2 = dict()
    
    # Diccionario con datos
    persona = {
        "nombre": "Ana",
        "edad": 20,
        "ciudad": "Madrid"
    }
    
    print(f"Diccionario vacío: {vacio}")
    print(f"Persona: {persona}")
    print(f"Tipo: {type(persona)}")
    print()
    
    # Acceso a valores
    print(f"Nombre: {persona['nombre']}")
    print(f"Edad: {persona['edad']}")
    print()
    
    # Añadir/modificar elementos
    persona["email"] = "ana@example.com"  # Añadir nueva clave
    persona["edad"] = 21  # Modificar valor existente
    print(f"Después de añadir email y modificar edad: {persona}")
    print()


def acceso_seguro():
    """Métodos seguros para acceder a valores."""
    print("=" * 60)
    print("ACCESO SEGURO A DICCIONARIOS")
    print("=" * 60)
    
    persona = {"nombre": "Carlos", "edad": 22}
    print(f"Diccionario: {persona}")
    print()
    
    # Acceso directo: lanza KeyError si no existe
    print(f"Nombre: {persona['nombre']}")
    try:
        print(persona["telefono"])  # Esto da error
    except KeyError as e:
        print(f"KeyError al acceder a 'telefono': {e}")
    print()
    
    # Método get: seguro, devuelve None si no existe
    nombre = persona.get("nombre")
    telefono = persona.get("telefono")
    print(f"get('nombre'): {nombre}")
    print(f"get('telefono'): {telefono}")
    print()
    
    # get con valor por defecto
    telefono = persona.get("telefono", "No disponible")
    ciudad = persona.get("ciudad", "Desconocida")
    print(f"get('telefono', 'No disponible'): {telefono}")
    print(f"get('ciudad', 'Desconocida'): {ciudad}")
    print()
    
    # Comprobar si existe una clave
    if "edad" in persona:
        print(f"La clave 'edad' existe: {persona['edad']}")
    
    if "telefono" not in persona:
        print("La clave 'telefono' NO existe")
    print()


def metodos_utiles():
    """Métodos importantes de diccionarios."""
    print("=" * 60)
    print("MÉTODOS ÚTILES DE DICCIONARIOS")
    print("=" * 60)
    
    notas = {
        "matemáticas": 8.5,
        "física": 7.0,
        "programación": 9.0
    }
    print(f"Diccionario original: {notas}")
    print()
    
    # keys(): obtener todas las claves
    claves = notas.keys()
    print(f"Claves: {claves}")
    print(f"Tipo: {type(claves)}")
    print(f"Como lista: {list(claves)}")
    print()
    
    # values(): obtener todos los valores
    valores = notas.values()
    print(f"Valores: {list(valores)}")
    print(f"Promedio: {sum(valores) / len(valores):.2f}")
    print()
    
    # items(): obtener pares clave-valor
    items = notas.items()
    print(f"Items: {list(items)}")
    print()
    
    # Iterar sobre diccionario
    print("Iterando sobre claves:")
    for asignatura in notas:
        print(f"  {asignatura}: {notas[asignatura]}")
    print()
    
    print("Iterando sobre items (mejor):")
    for asignatura, nota in notas.items():
        print(f"  {asignatura}: {nota}")
    print()


def modificar_diccionarios():
    """Añadir, modificar y eliminar elementos."""
    print("=" * 60)
    print("MODIFICAR DICCIONARIOS")
    print("=" * 60)
    
    inventario = {
        "manzanas": 10,
        "peras": 5
    }
    print(f"Inventario inicial: {inventario}")
    
    # Añadir
    inventario["naranjas"] = 8
    print(f"Después de añadir naranjas: {inventario}")
    
    # Modificar
    inventario["manzanas"] = 15
    print(f"Después de modificar manzanas: {inventario}")
    
    # Eliminar con del
    del inventario["peras"]
    print(f"Después de del inventario['peras']: {inventario}")
    
    # Eliminar con pop (devuelve el valor)
    valor = inventario.pop("naranjas")
    print(f"pop('naranjas') devolvió: {valor}")
    print(f"Inventario: {inventario}")
    
    # pop con valor por defecto (si no existe)
    valor = inventario.pop("plátanos", 0)
    print(f"pop('plátanos', 0) devolvió: {valor}")
    print()
    
    # update: fusionar diccionarios
    inventario = {"manzanas": 10, "peras": 5}
    nuevo_stock = {"naranjas": 8, "manzanas": 15}
    inventario.update(nuevo_stock)
    print(f"Después de update: {inventario}")
    print()
    
    # clear: vaciar diccionario
    copia = inventario.copy()
    copia.clear()
    print(f"Después de clear: {copia}")
    print(f"Original sin cambios: {inventario}")
    print()


def diccionarios_anidados():
    """Diccionarios dentro de diccionarios."""
    print("=" * 60)
    print("DICCIONARIOS ANIDADOS")
    print("=" * 60)
    
    # Base de datos de estudiantes
    estudiantes = {
        "12345A": {
            "nombre": "Ana García",
            "edad": 20,
            "notas": [8.5, 9.0, 7.5]
        },
        "67890B": {
            "nombre": "Carlos López",
            "edad": 22,
            "notas": [7.0, 8.0, 6.5]
        }
    }
    
    print("Base de datos de estudiantes:")
    for dni, datos in estudiantes.items():
        promedio = sum(datos["notas"]) / len(datos["notas"])
        print(f"  DNI: {dni}")
        print(f"    Nombre: {datos['nombre']}")
        print(f"    Edad: {datos['edad']}")
        print(f"    Promedio: {promedio:.2f}")
    print()
    
    # Acceder a datos anidados
    nombre_ana = estudiantes["12345A"]["nombre"]
    primera_nota_carlos = estudiantes["67890B"]["notas"][0]
    print(f"Nombre del estudiante 12345A: {nombre_ana}")
    print(f"Primera nota de 67890B: {primera_nota_carlos}")
    print()


def diccionarios_complejos():
    """Combinaciones de estructuras: listas de diccionarios, etc."""
    print("=" * 60)
    print("ESTRUCTURAS COMPLEJAS")
    print("=" * 60)
    
    # Lista de diccionarios (muy común)
    productos = [
        {"id": 1, "nombre": "Portátil", "precio": 899.99, "stock": 5},
        {"id": 2, "nombre": "Ratón", "precio": 19.99, "stock": 50},
        {"id": 3, "nombre": "Teclado", "precio": 49.99, "stock": 30}
    ]
    
    print("Catálogo de productos:")
    for producto in productos:
        print(f"  {producto['nombre']}: {producto['precio']}€ (stock: {producto['stock']})")
    print()
    
    # Buscar producto por nombre
    nombre_buscar = "Ratón"
    for producto in productos:
        if producto["nombre"] == nombre_buscar:
            print(f"Encontrado: {producto}")
            break
    print()
    
    # Calcular valor total del inventario
    valor_total = sum(p["precio"] * p["stock"] for p in productos)
    print(f"Valor total del inventario: {valor_total:.2f}€")
    print()


def comprehension_diccionarios():
    """Comprensión de diccionarios: crear diccionarios de forma compacta."""
    print("=" * 60)
    print("COMPRENSIÓN DE DICCIONARIOS")
    print("=" * 60)
    
    # Crear diccionario de cuadrados
    cuadrados = {x: x**2 for x in range(1, 6)}
    print(f"Cuadrados: {cuadrados}")
    
    # Invertir un diccionario (clave <-> valor)
    original = {"a": 1, "b": 2, "c": 3}
    invertido = {valor: clave for clave, valor in original.items()}
    print(f"Original: {original}")
    print(f"Invertido: {invertido}")
    
    # Con condición: solo pares
    numeros = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    pares = {k: v for k, v in numeros.items() if v % 2 == 0}
    print(f"Solo pares: {pares}")
    
    # Transformar valores
    precios_eur = {"pan": 1.5, "leche": 1.2, "huevos": 2.5}
    precios_cent = {prod: int(precio * 100) for prod, precio in precios_eur.items()}
    print(f"Precios en euros: {precios_eur}")
    print(f"Precios en céntimos: {precios_cent}")
    print()


def ejercicio_contador_palabras():
    """Ejercicio práctico: contar frecuencia de palabras."""
    print("=" * 60)
    print("EJERCICIO: CONTADOR DE PALABRAS")
    print("=" * 60)
    
    texto = "Python es genial Python es fácil Python es poderoso"
    palabras = texto.lower().split()
    print(f"Texto: '{texto}'")
    print(f"Palabras: {palabras}")
    print()
    
    # Contar frecuencias
    frecuencias = {}
    for palabra in palabras:
        if palabra in frecuencias:
            frecuencias[palabra] += 1
        else:
            frecuencias[palabra] = 1
    
    print("Frecuencias (forma tradicional):")
    for palabra, count in frecuencias.items():
        print(f"  '{palabra}': {count}")
    print()
    
    # Forma más pythónica con get
    frecuencias2 = {}
    for palabra in palabras:
        frecuencias2[palabra] = frecuencias2.get(palabra, 0) + 1
    
    print("Frecuencias (con get):")
    print(frecuencias2)
    print()


def ordenar_diccionarios():
    """Ordenar diccionarios por clave o por valor."""
    print("=" * 60)
    print("ORDENAR DICCIONARIOS")
    print("=" * 60)
    
    notas = {
        "Carlos": 7.0,
        "Ana": 9.5,
        "Beatriz": 8.0,
        "David": 6.5
    }
    print(f"Diccionario original: {notas}")
    print()
    
    # Ordenar por clave (alfabéticamente)
    ordenado_clave = dict(sorted(notas.items()))
    print(f"Ordenado por nombre: {ordenado_clave}")
    
    # Ordenar por valor (de menor a mayor nota)
    ordenado_valor = dict(sorted(notas.items(), key=lambda item: item[1]))
    print(f"Ordenado por nota (ascendente): {ordenado_valor}")
    
    # Ordenar por valor (de mayor a menor)
    ordenado_valor_desc = dict(sorted(notas.items(), key=lambda item: item[1], reverse=True))
    print(f"Ordenado por nota (descendente): {ordenado_valor_desc}")
    print()


if __name__ == "__main__":
    ejemplos_basicos()
    acceso_seguro()
    metodos_utiles()
    modificar_diccionarios()
    diccionarios_anidados()
    diccionarios_complejos()
    comprehension_diccionarios()
    ejercicio_contador_palabras()
    ordenar_diccionarios()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE DICCIONARIOS")
    print("=" * 60)
