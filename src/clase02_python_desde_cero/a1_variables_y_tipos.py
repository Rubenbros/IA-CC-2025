"""
Variables y tipos básicos en Python (comparado con Java)

- Python es de tipado dinámico: una variable puede referenciar valores de distintos tipos a lo largo del tiempo.
- Aun así, PEP 484 introduce anotaciones de tipo opcionales (type hints) para ayudar a herramientas como PyCharm.
- No uses `;` al final de las líneas. No hay tipos primitivos vs. envoltorios como en Java; todo es objeto.
- Palabras clave de valores especiales: `None` (equivalente aproximado a `null`), `True`, `False`.
- Convención de nombres: `snake_case` para variables y funciones; CONSTANTES en MAYÚSCULAS.
"""

from typing import Final

# Constante por convención (no es inmutable a nivel de lenguaje, pero lo tratamos como tal):
PI: Final[float] = 3.14159

if __name__ == "__main__":
    # Enteros (int) y flotantes (float)
    edad = 20        # int
    altura = 1.75    # float
    print("edad:", edad, "tipo:", type(edad))
    print("altura:", altura, "tipo:", type(altura))

    # Booleanos (bool)
    es_estudiante = True
    print("es_estudiante:", es_estudiante, "tipo:", type(es_estudiante))

    # None (~ null en Java)
    valor_desconocido = None
    print("valor_desconocido:", valor_desconocido, "tipo:", type(valor_desconocido))

    # Cadenas (str). En Python las usamos mucho y se manipulan fácilmente
    nombre = "Ada"
    saludo = f"Hola, {nombre}!"  # f-string: interpolación directa
    print(saludo)

    # Casting/conversiones explícitas
    numero_en_texto = "123"
    numero = int(numero_en_texto)  # de str a int
    print("numero:", numero, type(numero))

    texto = str(456)               # de int a str
    print("texto:", texto, type(texto))

    # Ojo: conversión a bool sigue la regla de "truthiness"
    # - 0, 0.0, cadena vacía "", contenedores vacíos ([], {}) y None son False
    # - todo lo demás es True
    print("bool(0):", bool(0))
    print("bool(1):", bool(1))
    print("bool(\"\"):", bool(""))
    print("bool(\"algo\"):", bool("algo"))

    # Tipado dinámico: la misma variable puede referenciar otro tipo después
    dato = 10        # int
    print("dato inicial:", dato, type(dato))
    dato = "diez"    # ahora str
    print("dato reasignado:", dato, type(dato))

    # Constantes: por convención no las cambiamos
    print("PI:", PI)
    # PI = 3.14  # NO lo hagas; aunque el lenguaje lo permite, va contra la convención
