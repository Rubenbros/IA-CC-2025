"""
Colecciones fundamentales en Python: list, tuple, set, dict (comparado con Java)

- list: secuencia mutable, ordenada. Similar a ArrayList en uso general.
- tuple: secuencia inmutable, ordenada. Útil para valores fijos o retornos múltiples.
- set: colección no ordenada de elementos únicos. Similar a HashSet.
- dict: mapeo clave→valor. Similar a HashMap.

- Los bucles `for` en Python iteran directamente sobre los elementos (como "for-each" en Java).
"""

if __name__ == "__main__":
    # LIST
    numeros = [10, 20, 30]
    numeros.append(40)          # añade al final
    numeros[1] = 25             # modifica (mutable)
    print("lista numeros:", numeros)

    # Iterar con for (tipo for-each)
    for n in numeros:
        print("n:", n)

    # Si necesitas el índice, usa enumerate
    for i, n in enumerate(numeros, start=1):
        print(f"{i}. {n}")

    # TUPLE
    punto = (2, 3)              # inmutable
    print("punto:", punto)
    # punto[0] = 5  # Error si lo descomentas (tuplas no se pueden modificar)

    # SET
    etiquetas = {"python", "java", "python"}  # los duplicados se eliminan
    etiquetas.add("datos")
    print("set etiquetas:", etiquetas)
    print("'java' en etiquetas?", "java" in etiquetas)

    # DICT (mapa clave→valor)
    persona = {"nombre": "Ada", "edad": 36}
    persona["ciudad"] = "Londres"  # añadir
    print("dict persona:", persona)

    # Recorrer claves y valores
    for clave, valor in persona.items():
        print(f"{clave} => {valor}")

    # Acceso seguro con get (devuelve None o un por defecto si no existe la clave)
    print("pais:", persona.get("pais"))
    print("pais (con por defecto):", persona.get("pais", "desconocido"))
