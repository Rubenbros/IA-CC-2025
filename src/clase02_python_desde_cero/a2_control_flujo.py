"""
Control de flujo en Python: if/elif/else y operadores lógicos (comparado con Java)

- Igual que en Java, tenemos `if`, `else if`, `else`; pero en Python se escribe `elif`.
- No hay paréntesis obligatorios alrededor de la condición.
- Los bloques se indican con `:` y se indentan con 4 espacios.
- Operadores: ==, !=, <, <=, >, >= y lógicos: and, or, not (en Java: &&, ||, !)
- La "verdad" en Python (truthiness): 0, 0.0, "", [], {}, None → se consideran False.
"""

if __name__ == "__main__":
    # Ejemplo 1: if/elif/else básico
    nota = 8  # cambia el valor para probar
    if nota >= 9:
        print("Sobresaliente")
    elif nota >= 7:
        print("Notable")
    elif nota >= 5:
        print("Aprobado")
    else:
        print("Suspendido")

    # Ejemplo 2: operadores lógicos (and, or, not)
    es_mayor_de_edad = True
    tiene_permiso = False

    if es_mayor_de_edad and tiene_permiso:
        print("Puedes entrar")
    elif es_mayor_de_edad and not tiene_permiso:
        print("Te falta permiso, pero eres mayor de edad")
    else:
        print("No puedes entrar")

    # Ejemplo 3: comparaciones encadenadas (Python permite esto)
    x = 5
    if 1 < x < 10:  # equivalente a (1 < x) and (x < 10)
        print("x está entre 1 y 10 (sin incluir)")

    # Nota: en Python usamos `==` para comparar valores; `is` es para identidad (mismo objeto)
    a = [1, 2]
    b = [1, 2]
    print("a == b:", a == b)  # True (mismo contenido)
    print("a is b:", a is b)  # False (objetos distintos en memoria)
