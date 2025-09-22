# Clase 01 — Introducción: Git, PyCharm y Python express

Esta guía contiene toda la información necesaria para impartir y seguir la primera sesión del curso.

Objetivos de la clase 01
- Configurar el entorno (Python + PyCharm + Git) y practicar el flujo básico con Git.
- Repasar Python básico para alumnado que viene de Java.
- Ejecutar y entender los ejercicios de ejemplo incluidos en el repositorio.

Contenido y código de esta clase
- Código principal: `src/clase01_intro_git_python`
- Requisitos opcionales: `requirements.txt` (puede estar vacío)
- Nota: Hemos eliminado tests automatizados para simplificar esta primera toma de contacto.

Descargas necesarias (enlaces oficiales)
- Python 3.11+ (Windows/macOS/Linux): https://www.python.org/downloads/
  - En Windows, durante la instalación marca “Add python.exe to PATH”.
- PyCharm Community (gratuito): https://www.jetbrains.com/pycharm/download/
  - Elige “Community Edition”. Abre el proyecto y crea el intérprete virtual (venv) desde Settings → Project → Python Interpreter.
- Git (Windows/macOS/Linux): https://git-scm.com/downloads
  - En Windows instala “Git for Windows”. Luego verifica con `git --version`.
- Opcional (GUI para Git): https://desktop.github.com/

Comprobar instalaciones (PowerShell o Terminal)
- `python --version`  (debería mostrar 3.11.x o similar)
- `pip --version`
- `git --version`

Preparación rápida del entorno
- Recomendado: Python 3.11+
- Opción A (PyCharm): abre el proyecto, crea un venv en Settings → Project → Python Interpreter → Add → Virtualenv. No es necesario instalar nada desde `requirements.txt` para esta clase inicial.
- Opción B (Terminal Windows PowerShell):
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1`

Cómo ejecutar los scripts de ejemplo
- Saludo: `python -m src.clase01_intro_git_python.hello Ada --formal --mayus`
- FizzBuzz: `python -m src.clase01_intro_git_python.fizzbuzz 20`
- POO Persona: `python -m src.clase01_intro_git_python.poo_persona`

Ejercicios de la clase (con verificación manual)
1) Saludo (funciones y argumentos)
- Archivo: `src/clase01_intro_git_python/hello.py`
- Objetivo: practicar funciones, parámetros por defecto y argumentos de línea de comandos.
- La función `greet(name, formal=False, mayus=False)` debe comportarse así:
  - `formal=True` → "Buenos días, Nombre" (sin exclamación)
  - `formal=False` → "Hola, Nombre!"
  - `mayus=True` → todo en MAYÚSCULAS
- Ejemplos esperados:
  - `greet("Ada")` → "Hola, Ada!"
  - `greet("Ada", formal=True)` → "Buenos días, Ada"
  - `greet("Ada", mayus=True)` → "HOLA, ADA!"

2) FizzBuzz (control de flujo y listas)
- Archivo: `src/clase01_intro_git_python/fizzbuzz.py`
- Objetivo: generar una lista de 1..n aplicando las reglas Fizz/Buzz.
- Verificación manual: para `n=5` deberías ver `["1", "2", "Fizz", "4", "Buzz"]`.
- Ejecución manual: `python -m src.clase01_intro_git_python.fizzbuzz 20`

3) POO Persona (clases y `__repr__`)
- Archivo: `src/clase01_intro_git_python/poo_persona.py`
- Objetivo: entender clases en Python (`self`, métodos de instancia, `__repr__`).
- Ampliación opcional:
  - Añade método `renombrar(nuevo_nombre: str)` que cambie el nombre y devuelva `self`.
  - Añade `presentarse(formal: bool=False)` que reutilice `saludar` y permita "Buenos días" si `formal`.

Consejos rápidos
- Indenta con 4 espacios. No mezcles tabs y espacios.
- Usa f-strings para formatear: `f"Hola, {name}!"`.
- Ejecuta y depura desde PyCharm para ver variables paso a paso.

Entrega sugerida
- Sube una captura de pantalla de cada script funcionando (salida en consola) y una breve nota explicando qué aprendiste.
- Si quieres, añade pequeños retos opcionales a cada ejercicio y súbelos en una rama con PR.

Organización de carpetas del curso (convención a seguir)
- Carpeta raíz para sesiones: `clases/`
- Una carpeta por sesión con el patrón: `NN-titulo-kebab-case`
  - `NN` es un número de dos dígitos (01, 02, 03, ...), para mantener orden cronológico.
  - `titulo-kebab-case` describe brevemente el tema de la sesión.

Estructura tipo por sesión (solo se crea cuando toque)

clases/
  NN-titulo/
    README.md              # guía de la sesión (objetivos, actividades, enlaces)
    apuntes/               # apuntes, guías rápidas, PDFs o MD (opcional)
    ejercicios/            # enunciados o notebooks específicos de la sesión (si no usan src/)
    datos/                 # datos pequeños para prácticas (si procede)
    soluciones/            # soluciones de los ejercicios (pueden publicarse más tarde)
    retos/                 # ejercicios extra/retos opcionales
    recursos/              # enlaces o materiales adicionales
    evaluacion/            # rúbricas, entregables o instrucciones de evaluación (si aplica)

Notas finales
- El código de Python de esta sesión está en `src/clase01_intro_git_python`.
- Si en futuras sesiones añadimos materiales específicos (notebooks, datasets…), se incluirán dentro de la carpeta de esa sesión siguiendo la estructura anterior.
- Evita espacios y acentos en nombres de carpetas/archivos. Usa `kebab-case` o `snake_case`.
- Próximas sesiones: seguiremos la convención creando `02-<tema>`, `03-<tema>`, etc. Si prefieres otra convención (por ejemplo `semana-01`), indícalo y lo adaptamos antes de crear más carpetas.
