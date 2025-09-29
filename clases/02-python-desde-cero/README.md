# Clase 02 — Conceptos teóricos de Python (guía breve)

Esta guía resume los conceptos fundamentales del lenguaje Python desde un punto de vista teórico. Está pensada como referencia rápida y neutral (sin comparativas con otros lenguajes ni instrucciones de ejecución).

## 1. Filosofía y diseño del lenguaje
- Legibilidad y sencillez por encima de la complejidad.
- “Hay una (y preferiblemente solo una) manera obvia de hacerlo” (Zen de Python).
- Indentación significativa: la estructura del programa se expresa con espacios, no con llaves.

## 2. Sintaxis base e indentación
- Bloques iniciados por `:` y definidos por la indentación (convención: 4 espacios).
- Instrucciones suelen ocupar una línea; no se usa `;` al final.
- Comentarios con `#`; cadenas multilínea también sirven como docstrings ("""...").

## 3. Modelo de objetos y tipado
- Todo es objeto (números, funciones, clases, módulos...).
- Tipado dinámico: el tipo está en el valor, no en el nombre de la variable.
- Tipado fuerte: no hay conversiones implícitas inseguros (p. ej., str + int falla).
- Identidad (quién es), igualdad (qué valor tiene) y tipo son conceptos distintos.

## 4. Tipos de datos fundamentales
- Numéricos: `int` (precisión arbitraria), `float` (doble precisión), `complex`.
- Booleano: `bool` con valores `True`/`False`.
- Texto: `str` (Unicode); inmutable.
- Secuencias: `list` (mutable, ordenada), `tuple` (inmutable, ordenada), `range`.
- Conjuntos: `set` (sin orden, elementos únicos), `frozenset` (inmutable).
- Mapeos: `dict` (clave→valor, mutable, sin orden garantizado por contrato, aunque mantiene inserción).
- Nulo/ausente: `None` (un único objeto que representa “no hay valor”).
- Inmutabilidad: `str`, `tuple`, `frozenset` son inmutables; `list`, `dict`, `set` son mutables.

## 5. Truthiness (verdad por conveniencia)
- Se consideran falsos: `False`, `None`, `0`/`0.0`, `0j`, `""`, contenedores vacíos (`[]`, `{}`, `set()`, `()`).
- Todo lo demás se evalúa como verdadero en contextos booleanos.

## 6. Operadores y expresiones
- Comparación: `==`, `!=`, `<`, `<=`, `>`, `>=`.
- Lógicos: `and`, `or`, `not` (evalúan corto-circuito).
- Pertenencia e identidad: `in`/`not in`, `is`/`is not`.
- Encadenamiento de comparaciones: `a < b < c` evalúa de forma transitiva.
- Operadores sobre secuencias: concatenación `+`, repetición `*`, pertenencia `in`.

## 7. Control de flujo
- Selección: `if` / `elif` / `else`.
- Iteración:
  - `for` itera sobre elementos de un iterable.
  - `while` repite mientras la condición sea verdadera.
- Interrupciones: `break` (sale del bucle), `continue` (siguiente iteración), `pass` (no-op).
- Clausula `else` en bucles: se ejecuta si el bucle termina sin `break`.

## 8. Funciones (definición y semántica)
- Definición: `def nombre(parámetros):` con bloque indentado.
- Docstrings: la primera cadena del cuerpo documenta la función.
- Parámetros:
  - Posicionales y por palabra clave (keyword-only con `*`).
  - Valores por defecto evaluados en tiempo de definición (¡cuidado con mutables!).
  - Variádicos: `*args` (tupla de posicionales), `**kwargs` (dict de nombrados).
- Retorno: `return` (si se omite, devuelve `None`).
- Funciones son ciudadanos de primera clase: se pueden pasar como valores.

## 9. Ámbito de nombres (LEGB)
- Resolución de nombres: Local → Enclosing (no local) → Global (módulo) → Builtins.
- `global` y `nonlocal` permiten enlazar a nombres de ámbitos externos específicos.

## 10. Módulos y paquetes
- Módulo: archivo `.py` que define un espacio de nombres propio.
- Paquete: carpeta con `__init__.py` que agrupa módulos y subpaquetes.
- Importación: crea una única instancia del módulo por proceso; se cachea en `sys.modules`.
- Atributo `__name__`: nombre cualificado del módulo; sirve para detectar contexto de importación.

## 11. Manejo de errores (excepciones)
- Excepciones son objetos que interrumpen el flujo normal; se capturan con `try/except`.
- Clausulas:
  - `try/except` (captura), `try/except/else` (código si no hubo excepción), `try/finally` (limpieza), o combinadas.
- Propagación: si no se captura, la excepción burbujea hasta abortar el programa.
- Uso de `raise` para señalar condiciones inválidas o violaciones de contrato.

## 12. Objetos y clases (nociones básicas)
- Clase define un tipo; instancia es un objeto de esa clase.
- Métodos de instancia reciben `self` como primer parámetro por convención.
- Atributos pueden definirse en la clase (compartidos) o en la instancia.
- Métodos estáticos y de clase: `@staticmethod` y `@classmethod`.
- Modelo de herencia simple; resolución de atributos vía MRO (orden de resolución de métodos).

## 13. Anotaciones de tipo (type hints)
- Sintaxis PEP 484: opcionales, no afectan la ejecución en CPython.
- Útiles para herramientas (editores, linters, analizadores estáticos).
- Ejemplos conceptuales: `x: int`, `def f(a: str) -> bool: ...`, `Optional[T]`, `Union[...]`, `Iterable[T]`.

## 14. Estilo y convenciones (PEP 8)
- Nombres en `snake_case` para funciones/variables; `CapWords` para clases; CONSTANTES en mayúsculas.
- 4 espacios por nivel de indentación; líneas de hasta ~79–99 columnas como guía.
- Espacios alrededor de operadores, después de coma/puntos y coma.
- Docstrings con triple comilla; preferir f-strings para formateo de texto.

## 15. Modelo de ejecución e importación
- Código en el “nivel superior” del módulo se ejecuta al importar.
- Mantener lógica en funciones/clases y limitar efectos colaterales al importar.
- La importación es idempotente por proceso (módulos se cachean); para re-ejecutar hay que recargar explícitamente.

## 16. Glosario mínimo
- Objeto: entidad con identidad, tipo y valor.
- Iterable/Iterador: protocolo para producir elementos (iterables crean iteradores).
- Inmutable/Mutable: si su estado puede cambiar tras crearse.
- Duck typing: énfasis en el comportamiento más que en el tipo nominal.

---