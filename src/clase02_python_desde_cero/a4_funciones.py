"""
Funciones en Python (comparado con Java)

- Se definen con `def nombre(parametros):` y un bloque indentado.
- Las anotaciones de tipo son opcionales: `def sumar(a: int, b: int) -> int:` ayudan a herramientas.
- Retorno con `return` (puede devolver cualquier cosa; incluso múltiples valores como tupla).
- Parámetros por defecto y argumentos con nombre (keyword args) hacen las llamadas muy legibles.
- No existe sobrecarga por firma como en Java. Se usan valores por defecto o *args/**kwargs.
"""

from typing import Optional, Tuple


def sumar(a: int, b: int) -> int:
    """Devuelve la suma de a + b."""
    return a + b


def presentar(nombre: str, formal: bool = False) -> str:
    """Devuelve un saludo. Si `formal` es True, usa "Buenos días".

    - En Java podrías tener dos métodos sobrecargados; en Python preferimos un parámetro opcional.
    """
    if formal:
        return f"Buenos días, {nombre}"
    return f"Hola, {nombre}!"


def dividir(a: float, b: float) -> Optional[float]:
    """Divide a entre b. Si b es 0, devuelve None (en vez de lanzar excepción)."""
    if b == 0:
        return None
    return a / b


def coords() -> Tuple[int, int]:
    """Devuelve dos valores (una tupla). En la llamada se puede desempaquetar."""
    return 2, 3


if __name__ == "__main__":
    # Llamadas simples
    print("sumar(2, 3) =", sumar(2, 3))

    # Parámetros con nombre (keyword args):
    print(presentar(nombre="Ada"))
    print(presentar(nombre="Ada", formal=True))

    # Manejo de retorno opcional (None)
    resultado = dividir(10, 0)
    if resultado is None:
        print("División por cero: resultado None")
    else:
        print("Resultado división:", resultado)

    # Retornos múltiples (tuplas) y desempaquetado
    x, y = coords()
    print("coords -> x:", x, "y:", y)
