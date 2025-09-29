# Tarea para casa — Clase 02 (Python desde 0, alumnado que viene de Java)

Objetivo: consolidar lo visto en la clase 02 trabajando funciones, control de flujo, colecciones y manejo básico de excepciones. Los ejercicios están pensados para practicarlos sin librerías externas.

Entrega: 7 días desde la sesión

Requisitos técnicos
- Python 3.11+
- No uses librerías externas (solo librería estándar si hace falta: `statistics`, `string`, etc.).
- Evita `input()` en los ejercicios: define funciones puras y, si quieres demostrar resultados, haz `print()` dentro de un bloque `if __name__ == "__main__":`.
- Estilo: `snake_case`, f-strings, docstrings breves y type hints donde te resulte natural.

Forma de entrega (elige una)
- Opción A (carpeta en el repo de clase): crea `entregas\\clase02\\<tu-nombre>\\` y coloca uno o varios `.py` (por ejemplo `tarea_clase02.py`).
- 
Consejo para quienes vienen de Java
- Piensa en pequeñas funciones puras con entradas y salidas claras en vez de clases obligatorias. Usa pruebas rápidas con prints.

---

## Bloque 1 — Básico
1) Conversión de temperaturas
- Implementa `c_to_f(c: float) -> float` y `f_to_c(f: float) -> float`.
- Redondea a 2 decimales en el retorno.
- Ejemplos esperados: `c_to_f(0) -> 32.0`, `f_to_c(212) -> 100.0`.

2) Normalización de nombres
- `normaliza_nombre(nombre: str) -> str` que elimine espacios extra y capitalice cada palabra.
- Ejemplos: `"  aDa   lovelace " -> "Ada Lovelace"`, `"alan turing" -> "Alan Turing"`.

3) Clasificador de notas (versión función)
- `clasificar_nota(nota: float) -> str` devolviendo: "Suspendido" (<5), "Aprobado" (>=5), "Notable" (>=7), "Sobresaliente" (>=9).
- Nota: prioriza los mayores primero (>=9) o usa `if/elif` ordenado correctamente.

4) Contador de letras
- `contar_letras(texto: str) -> dict[str, int]` que devuelva número de vocales y consonantes, ignorando espacios, dígitos y signos.
- Sugerencia: normaliza a minúsculas, considera vocales `aeiou`. Puedes usar `str.isalpha()`.
- Resultado tipo: `{"vocales": 7, "consonantes": 12}`.

5) FizzBuzz parametrizable
- `fizzbuzz(n: int, tres: str = "Fizz", cinco: str = "Buzz") -> list[str]`.
- Para números múltiplos de 3 usa `tres`, de 5 usa `cinco`, de ambos concaténalos.
- Ejemplo: `fizzbuzz(5) -> ["1", "2", "Fizz", "4", "Buzz"]`.

---

## Bloque 2 — Intermedio
6) Estadísticas de lista
- `estadisticas(nums: list[float]) -> dict[str, float]` con `min`, `max`, `media` y `mediana`.
- Si la lista está vacía, lanza `ValueError` con un mensaje claro.
- Puedes usar `statistics.mean` y `statistics.median` o implementarlas.

7) Validador de contraseñas
- `validar_password(pw: str) -> tuple[bool, list[str]]`.
- Reglas mínimas: longitud >= 8, al menos una mayúscula, una minúscula, un dígito y un símbolo (`!@#$%^&*` u otros).
- Devuelve `(True, [])` si pasa; si no, `(False, ["falta mayúscula", "longitud < 8", ...])`.

8) Inventario sencillo
- Dadas líneas de compra `list[tuple[str, int]]` (producto, cantidad), implementa `acumular_inventario(lineas) -> dict[str, int]`.
- Suma cantidades por producto; si alguna queda <= 0, elimínala del resultado.
- Ejemplo: `[("manzana", 2), ("pera", 1), ("manzana", 3)] -> {"manzana": 5, "pera": 1}`.

9) Agenda y búsqueda por prefijo
- `cargar_agenda(texto: str) -> dict[str, str]` donde `texto` tiene líneas `Nombre:telefono`.
- `buscar_por_prefijo(agenda: dict[str, str], prefijo: str) -> dict[str, str]`.
- Ignora líneas vacías y nombres repetidos (el último gana).

---

## Reto
10) Adivina el número (sin `input()`).
- Implementa `pensar_numero(seed: int | None = None) -> int` que use `random` para generar un número 1..100.
- Implementa `intentos_para_acertar(objetivo: int, estrategia) -> int` donde `estrategia` es una función que, dado un rango, devuelve la siguiente conjetura (p.ej. búsqueda binaria). Cuenta los intentos hasta acertar.
- Demuestra en `__main__` que una estrategia binaria acierta en <= 7 intentos para cualquier 1..100.

---

## Verificación manual (ejemplo de uso)
Crea un archivo `tarea_clase02.py` y añade al final algo como:

```python
if __name__ == "__main__":
    print("c_to_f(0)", c_to_f(0))
    print("normaliza_nombre('  aDa   lovelace ')", normaliza_nombre("  aDa   lovelace "))
    print("clasificar_nota(8.5)", clasificar_nota(8.5))
    print("fizzbuzz(15)", fizzbuzz(15))
    try:
        print(estadisticas([1, 2, 3, 4]))
    except ValueError as e:
        print("Error:", e)
```

Esto no sustituye tests; solo te ayuda a ver salidas rápidas.

## Entrega
- Sube tus `.py` a `entregas\\clase02\\<tu-nombre>\\` o abre un PR con rama `tarea/clase02/<tu-nombre>`.
- Incluye un README corto con notas si hiciste decisiones de diseño.

¡Listo! Céntrate en funciones pequeñas y claras. Si algo se complica, escribe primero ejemplos esperados como comentarios y luego implementa.


## Guía: Cómo crear un Pull Request desde PyCharm (con tus ejercicios)

Sigue estos pasos para entregar tus ejercicios mediante un Pull Request (PR) directamente desde PyCharm. Están pensados para Windows, pero son muy parecidos en macOS/Linux.

Pre-requisitos
- Tener el repositorio del curso clonado en tu máquina y abierto en PyCharm.
- Recomendado: trabajar desde tu fork en GitHub (origin = tu fork) y añadir el remoto upstream (el repo original del curso) para abrir el PR contra él.

1) Crea la rama de trabajo
- En PyCharm: menú Git → New Branch...
- Nombre sugerido: tarea/clase02/<tu-nombre> (sustituye <tu-nombre> por algo sin espacios, ej. "rosa-jarne").
- Marca "Checkout branch" para cambiarte a esa rama.

2) Crea la carpeta de entrega y añade tus archivos
- En el panel del proyecto, crea la carpeta: entregas\clase02\<tu-nombre>\
- Dentro, añade tu(s) archivo(s) .py, por ejemplo: entregas\clase02\<tu-nombre>\tarea_clase02.py
- Opcional: añade un README.md corto con notas de diseño o decisiones que tomaste.

3) Haz commit de tus cambios
- Abre la ventana Git → Commit (o atajo desde la barra inferior: Commit).
- Selecciona los archivos nuevos/cambiados, escribe un mensaje claro, ej.: "tarea(clase02): ejercicios y README de <tu-nombre>".
- Pulsa Commit (o Commit and Push si ya quieres subirlos).

4) Configura remotos (solo si no los tienes)
- Git → Manage Remotes...
- Debes ver:
  - origin → tu fork en GitHub (tu-usuario/IA-CC-2025)
  - upstream → repo original del curso (profesor/IA-CC-2025)
- Si falta upstream, añádelo con la URL del repo original. Así podrás abrir el PR "hacia" upstream.

5) Sube la rama al remoto
- Git → Push...
- Elige subir la rama tarea/clase02/<tu-nombre> a origin.
- Acepta el diálogo. PyCharm creará la rama remota si no existe.

6) Crea el Pull Request desde PyCharm
- Tras el push, PyCharm suele mostrar un botón o enlace "Create Pull Request".
- Si lo ves: haz clic y rellena:
  - Base repository: upstream (el repo original del curso).
  - Base branch: main (o la rama indicada por el docente).
  - Head repository: tu fork (origin).
  - Head branch: tarea/clase02/<tu-nombre>.
  - Título sugerido: "tarea: clase02 — <tu-nombre>".
  - Descripción: indica qué ejercicios incluyes, decisiones tomadas, notas de verificación.
- Si no aparece el botón (o usas PyCharm Community sin integración):
  - Abre GitHub en el navegador: ve a tu fork → verás un aviso para "Compare & pull request". Haz clic y rellena los mismos campos.

7) Verifica el PR
- Revisa el diff: comprueba que solo van tus archivos/cambios.
- Asegúrate de que la ruta de entrega es la correcta: entregas/clase02/<tu-nombre>/...
- Envía el PR. Puedes añadir reviewers si el docente lo pide.

8) Itera si hay feedback
- Si te piden cambios, edita en la misma rama (tarea/clase02/<tu-nombre>), haz commit y push. El PR se actualiza automáticamente.

Consejos y resolución de problemas
- Nombre de rama: evita espacios y acentos. Usa letras minúsculas, guiones y/o barras.
- Usuario sin fork: si no tienes permiso para push en el repo original, crea un fork (botón Fork en GitHub) y clona tu fork en local antes de empezar.
- Remotos confundidos: si tu origin apunta al repo del profe, clona tu fork y vuelve a empezar, o cambia origin→fork y añade upstream→repo original.
- Autenticación: si Git pide usuario/contraseña, usa un token personal (PAT) de GitHub o el helper de credenciales que PyCharm configura.
- Mensajes de commit: usa un prefijo coherente, ej. "tarea(clase02): ...".

Ejemplo rápido de estructura final
- entregas/
  - clase02/
    - <tu-nombre>/
      - tarea_clase02.py
      - README.md (opcional)

Con esto, tu entrega por PR desde PyCharm queda documentada. ¡Éxitos!