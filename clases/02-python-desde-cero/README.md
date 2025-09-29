# Clase 02 — Python desde 0 (para alumnado que viene de Java)

Esta guía introduce Python desde cero con explicaciones pensadas para quien ya ha visto Java. Iremos paso a paso y con muchos comentarios en el código.

Objetivos de la clase 02
- Entender cómo se ejecuta un script en Python (sin clases obligatorias ni `main` estático).
- Conocer tipos básicos, variables y reglas de estilo (PEP8).
- Practicar control de flujo: `if/elif/else`, bucles `for` y `while` básicos.
- Usar colecciones fundamentales: `list`, `tuple`, `set`, `dict`.
- Definir funciones, parámetros por defecto y argumentos con nombre.
- Manejar excepciones y comprender diferencias clave con Java.

Código y estructura de esta clase
- Paquete de ejemplo: `src/clase02_python_desde_cero`
- Ejecuta cada archivo con: `python -m src.clase02_python_desde_cero.<modulo>`
  - Ejemplo: `python -m src.clase02_python_desde_cero.a0_hola_mundo`

Requisitos previos (mismos que la clase 01)
- Python 3.11+ y un entorno (PyCharm recomendado).
- No es necesario instalar dependencias extra para esta sesión.

Ruta sugerida y scripts (en orden)
1) Hola Mundo y estructura mínima
   - `a0_hola_mundo.py`: muestra `print`, comentarios, indentación y el bloque `if __name__ == "__main__":` (equivalente al `main` en Java).

2) Variables y tipos básicos
   - `a1_variables_y_tipos.py`: enteros, flotantes, booleanos, `None`, conversiones de tipo, `type()`, constantes por convención.

3) Control de flujo
   - `a2_control_flujo.py`: `if/elif/else`, operadores lógicos, verdad/falsez en Python.

4) Colecciones
   - `a3_colecciones.py`: `list`, `tuple`, `set`, `dict`, iteración con `for`.

5) Funciones
   - `a4_funciones.py`: `def`, argumentos por defecto, argumentos con nombre, valores de retorno.

6) Excepciones
   - `a5_excepciones.py`: `try/except`, captura de errores comunes, `finally`.

Anexo: diferencias clave con Java
- `b0_diferencias_con_java.py`: resumen comentado para tener a mano cuando programes.

Cómo ejecutar ejemplos (Terminal o consola de PyCharm)
- `python -m src.clase02_python_desde_cero.a0_hola_mundo`
- `python -m src.clase02_python_desde_cero.a1_variables_y_tipos`
- `python -m src.clase02_python_desde_cero.a2_control_flujo`
- `python -m src.clase02_python_desde_cero.a3_colecciones`
- `python -m src.clase02_python_desde_cero.a4_funciones`
- `python -m src.clase02_python_desde_cero.a5_excepciones`
- `python -m src.clase02_python_desde_cero.b0_diferencias_con_java`

Ejercicios propuestos (verificación manual)
1) Saludo personalizado ampliado
   - En `a4_funciones.py`, crea una función `saludar(nombre: str, formal: bool=False)` que devuelva:
     - Formal: "Buenos días, Nombre"
     - No formal: "Hola, Nombre!"
   - Pruébala llamándola con distintos argumentos (con nombre y posicionales) y haciendo `print` del resultado.

2) Clasificador de notas
   - En `a2_control_flujo.py`, dada una nota numérica 0..10, imprime: Suspendido (<5), Aprobado (>=5), Notable (>=7), Sobresaliente (>=9).
   - Pruébalo con varios valores.

3) Lista de tareas
   - En `a3_colecciones.py`, crea una lista `tareas = ["instalar Python", "abrir PyCharm", "executar script"]`.
   - Añade una nueva tarea con `append`, recórrela con `for` e imprime cada una con un índice (empezando en 1).

Consejos rápidos
- Indentación: 4 espacios por nivel (sin llaves `{}`); la indentación marca los bloques.
- Estilo: usa `snake_case` para variables y funciones. Constantes en mayúsculas (`MI_CONSTANTE`).
- Formato de cadenas: usa f-strings: `f"Hola, {nombre}!"`.
- Tipado: las anotaciones de tipo (`: str`, `-> int`) son opcionales pero recomendables.

Si algo no funciona
- Ejecuta desde la raíz del proyecto y revisa que la ruta del módulo sea correcta.
- Comprueba la versión de Python: `python --version`.
- Abre los archivos en PyCharm, lee los comentarios y ejecuta con el botón ▶ del editor.


Tarea para casa (deberes)
- Objetivo: consolidar variables y tipos, control de flujo, colecciones, funciones y excepciones con ejercicios graduados.
- Entrega y rúbrica: ver el archivo detallado en ejercicios/tarea-casa.md.
- Requisitos: Python 3.11+, sin librerías externas. Preferir funciones puras y verificación manual con prints en __main__.
- Enlace directo: [ejercicios/tarea-casa.md](ejercicios/tarea-casa.md)
