"""
Ejemplo de control de flujo clásico: FizzBuzz.
- Para 1..n imprime:
  * "Fizz" si divisible por 3
  * "Buzz" si divisible por 5
  * "FizzBuzz" si divisible por ambos
  * en otro caso el propio número

Notas para quienes vienen de Java:
- range(1, n+1) genera 1..n (el límite superior es exclusivo, por eso sumamos 1).
- El operador resto es % como en Java.
- Usamos list[str] como anotación de tipo (Python 3.9+). En versiones antiguas sería List[str].
"""

from typing import List


def fizzbuzz(n: int) -> list[str]:
    """Devuelve una lista de strings con el resultado de FizzBuzz de 1 a n."""
    resultado: list[str] = []
    for i in range(1, n + 1):  # en Java: for (int i = 1; i <= n; i++) { ... }
        salida = ""
        if i % 3 == 0:
            salida += "Fizz"
        if i % 5 == 0:
            salida += "Buzz"
        if salida == "":
            salida = str(i)  # convertimos int → str para homogeneidad del tipo
        resultado.append(salida)
    return resultado


if __name__ == "__main__":
    # Ejecuta: python -m src.clase01_intro_git_python.fizzbuzz 20
    import argparse

    parser = argparse.ArgumentParser(description="FizzBuzz 1..n")
    parser.add_argument("n", type=int, help="Límite superior (entero positivo)")
    args = parser.parse_args()

    print(", ".join(fizzbuzz(args.n)))
