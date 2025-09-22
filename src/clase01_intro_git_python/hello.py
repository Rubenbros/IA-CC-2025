"""
Demostración sencilla para quienes vienen de Java.

- En Java tendrías una clase con un método static main(String[] args).
- En Python, la entrada típica es un bloque if __name__ == "__main__": al final del archivo.
- print(...) equivale a System.out.println(...).
- Los tipos (str, bool) son "type hints" opcionales: ayudan a PyCharm y a linters, pero
  Python no los obliga en tiempo de ejecución.
"""

from typing import Final

DEFAULT_NAME: Final[str] = "mundo"  # Constante (por convención, mayúsculas; no es realmente inmutable)


def greet(name: str, formal: bool = False, mayus: bool = False) -> str:
    """Devuelve un saludo para "name".

    Parámetros
    - name: nombre a saludar (str en vez de String)
    - formal: si True, usa un saludo formal (similar a pasar un flag booleano en Java)
    - mayus: si True, devuelve el texto en MAYÚSCULAS

    Nota mental Java → Python:
    - No hay operador ternario "? :". En Python usamos A if cond else B.
    - Las cadenas se interpolan con f-strings: f"Hola, {name}!".
    """
    saludo = f"Buenos días, {name}" if formal else f"Hola, {name}!"
    return saludo.upper() if mayus else saludo


if __name__ == "__main__":
    # Punto de entrada del script (equivalente al main de Java).
    # Ejecuta desde PyCharm con el botón ▶ o desde terminal: python -m src.clase01_intro_git_python.hello Ada --formal
    import argparse  # Librería estándar para parsear argumentos (similar a args[] y librerías CLI en Java)

    parser = argparse.ArgumentParser(description="Demo de saludo para Java → Python")
    parser.add_argument("name", nargs="?", default=DEFAULT_NAME, help="Nombre a saludar (por defecto: mundo)")
    parser.add_argument("--formal", action="store_true", help="Usar saludo formal (Buenos días)")
    parser.add_argument("--mayus", action="store_true", help="Convertir a MAYÚSCULAS")
    args = parser.parse_args()

    # En Java: System.out.println(...)
    print(greet(args.name, formal=args.formal, mayus=args.mayus))
