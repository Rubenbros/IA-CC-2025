"""
Hola Mundo — el programa más pequeño en Python.

Si vienes de Java:
- En Java sueles tener una clase pública con un `public static void main(String[] args)`.
- En Python NO necesitas crear una clase para empezar. Un archivo `.py` ya es un módulo ejecutable.
- Para marcar el punto de entrada, solemos usar este patrón al final del archivo:

    if __name__ == "__main__":
        # código a ejecutar cuando se corre el archivo como script

- `print(...)` en Python es como `System.out.println(...)` en Java.
- No hay `;` al final de la línea.
- Los bloques se marcan por indentación (4 espacios), no por llaves `{}`.
"""

# OJO (especialmente si vienes de Java): en Python, cualquier código que esté
# en el "nivel superior" del archivo (fuera de funciones/clases y fuera del
# bloque if __name__ == "__main__") se ejecuta en cuanto el módulo se carga,
# ya sea porque lo ejecutas como script o porque lo importas desde otro archivo.
# Por eso esta línea se ejecuta aunque no esté dentro de __main__.
# Si NO quieres que se ejecute al importar, muévela dentro del bloque __main__.
# Una sola línea que imprime un texto en la consola:
print("Hola, mundo")  # ← se ejecuta al cargar el módulo (nivel superior)


# Punto de entrada del script: similar a tener un main en Java
if __name__ == "__main__":
    # Este bloque se ejecuta solo cuando corres:
    #   python -m src.clase02_python_desde_cero.a0_hola_mundo
    # (Es equivalente a ejecutar este archivo como script)

    # En Java harías: System.out.println("Ejecutado como script");
    print("Ejecutado como script — __name__ == '__main__'")
