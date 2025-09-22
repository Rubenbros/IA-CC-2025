"""
POO en Python comparada con Java.

- No hay palabra clave "class Persona { ... }" con llaves; los bloques se delimitan por indentación.
- Los métodos reciben "self" como primer parámetro (similar a this en Java, pero explícito).
- No existen campos privados estrictos; por convención se usa _prefijo para indicar "interno".
- No hay "static" por defecto. Se puede usar @staticmethod o @classmethod si hace falta.
"""

class Persona:
    def __init__(self, nombre: str):
        # En Java escribiríamos: this.nombre = nombre; aquí es similar
        self.nombre = nombre

    def saludar(self) -> str:
        return f"Hola, soy {self.nombre}"

    def __repr__(self) -> str:
        # Representación útil para depuración (equivalente a toString, pero técnica distinta)
        return f"Persona(nombre={self.nombre!r})"


if __name__ == "__main__":
    # Ejecuta este archivo directamente: python -m src.clase01_intro_git_python.poo_persona
    ada = Persona("Ada")

    # En Java: System.out.println(ada.toString());
    print(ada)  # llama a __repr__
    print(ada.saludar())
