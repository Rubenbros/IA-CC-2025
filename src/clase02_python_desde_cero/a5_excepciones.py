"""
Excepciones básicas en Python (comparado con Java)

- try/except en Python captura excepciones (similar a try/catch en Java).
- Puedes capturar excepciones específicas (ValueError, ZeroDivisionError, etc.).
- finally se ejecuta siempre (como en Java), útil para liberar recursos.
- raise lanza una excepción.
"""

if __name__ == "__main__":
    # Ejemplo 1: capturar ValueError al convertir texto a int
    textos = ["10", "abc", "25"]
    for t in textos:
        try:
            numero = int(t)
            print(f"'{t}' -> {numero}")
        except ValueError as e:
            print(f"No se pudo convertir '{t}' a int:", e)
        finally:
            # Se ejecuta siempre
            pass

    # Ejemplo 2: ZeroDivisionError
    a, b = 10, 0
    try:
        print(a / b)
    except ZeroDivisionError:
        print("Error: división entre cero")

    # Ejemplo 3: lanzar nuestra propia excepción
    def raiz_cuadrada(x: float) -> float:
        if x < 0:
            raise ValueError("x debe ser >= 0")
        return x ** 0.5

    try:
        print("raiz_cuadrada(9) ->", raiz_cuadrada(9))
        print("raiz_cuadrada(-1) ->", raiz_cuadrada(-1))
    except ValueError as err:
        print("Fallo en raiz_cuadrada:", err)
