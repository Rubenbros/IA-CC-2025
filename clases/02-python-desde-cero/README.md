# Clase 02 — Conceptos teóricos de Python (guía breve)

Esta guía resume los conceptos fundamentales del lenguaje Python desde un punto de vista teórico. Está pensada como referencia rápida y neutral (sin comparativas con otros lenguajes ni instrucciones de ejecución).

## 1. Filosofía y diseño del lenguaje
- Legibilidad y sencillez por encima de la complejidad.
- “Hay una (y preferiblemente solo una) manera obvia de hacerlo” (Zen de Python).
- Indentación significativa: la estructura del programa se expresa con espacios, no con llaves.

## 2. Sintaxis base e indentación
- Bloques iniciados por `:` y definidos por la indentación (convención: 4 espacios).
- Instrucciones suelen ocupar una línea; no se usa `;` al final.
- Comentarios con `#`; cadenas multilínea también sirven como docstrings ("""...).

```python
# Comentario de una línea
def saludar(nombre):  # Los dos puntos inician el bloque
    """Docstring que documenta la función."""  # Triple comillas
    if nombre:  # Indentación de 4 espacios
        print(f"Hola, {nombre}!")  # Instrucción dentro del bloque
        return True
    else:
        print("No hay nombre")
        return False

# Múltiples instrucciones en una línea (no recomendado)
x = 1; y = 2; print(x, y)
```

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
- Nulo/ausente: `None` (un único objeto que representa "no hay valor").
- Inmutabilidad: `str`, `tuple`, `frozenset` son inmutables; `list`, `dict`, `set` son mutables.

```python
# Numéricos
entero = 42
grande = 12345678901234567890  # Precisión arbitraria
decimal = 3.14
complejo = 3 + 4j

# Booleano
verdadero = True
falso = False

# Texto (inmutable)
texto = "Hola mundo"
unicode_text = "Café ☕"

# Listas (mutables)
numeros = [1, 2, 3, 4]
mixta = [1, "dos", 3.0, True]
numeros.append(5)  # Modifica la lista

# Tuplas (inmutables)
coordenadas = (10, 20)
punto = (3.5, 2.8, "A")

# Rangos
rango = range(0, 10, 2)  # 0, 2, 4, 6, 8

# Conjuntos
unicos = {1, 2, 3, 3, 2}  # {1, 2, 3}
inmutable_set = frozenset([1, 2, 3])

# Diccionarios
persona = {"nombre": "Ana", "edad": 30, "activo": True}
persona["ciudad"] = "Madrid"  # Se puede modificar

# Nulo
vacio = None
```

## 5. Truthiness (verdad por conveniencia)
- Se consideran falsos: `False`, `None`, `0`/`0.0`, `0j`, `""`, contenedores vacíos (`[]`, `{}`, `set()`, `()`).
- Todo lo demás se evalúa como verdadero en contextos booleanos.

```python
# Valores falsos (falsy)
falsy_values = [
    False,          # Booleano falso
    None,           # Valor nulo
    0,              # Entero cero
    0.0,            # Float cero
    0j,             # Complejo cero
    "",             # String vacío
    [],             # Lista vacía
    {},             # Diccionario vacío
    set(),          # Conjunto vacío
    (),             # Tupla vacía
]

# Valores verdaderos (truthy)
truthy_values = [
    True,           # Booleano verdadero
    1,              # Cualquier número no cero
    "texto",        # String no vacío
    [1, 2],         # Lista con elementos
    {"a": 1},       # Diccionario con elementos
    {1, 2},         # Conjunto con elementos
]

# Uso en contextos booleanos
def evaluar_truthiness(valor):
    if valor:
        print(f"{valor} es truthy")
    else:
        print(f"{valor} es falsy")

# Ejemplos
evaluar_truthiness("")       # "" es falsy
evaluar_truthiness("hola")   # "hola" es truthy
evaluar_truthiness([])       # [] es falsy
evaluar_truthiness([1])      # [1] es truthy
```

## 6. Operadores y expresiones
- Comparación: `==`, `!=`, `<`, `<=`, `>`, `>=`.
- Lógicos: `and`, `or`, `not` (evalúan corto-circuito).
- Pertenencia e identidad: `in`/`not in`, `is`/`is not`.
- Encadenamiento de comparaciones: `a < b < c` evalúa de forma transitiva.
- Operadores sobre secuencias: concatenación `+`, repetición `*`, pertenencia `in`.

```python
# Operadores de comparación
a, b = 10, 20
print(a == b)    # False
print(a != b)    # True
print(a < b)     # True
print(a <= 10)   # True

# Operadores lógicos (cortocircuito)
x = True
y = False
print(x and y)   # False
print(x or y)    # True
print(not x)     # False

# Cortocircuito: no evalúa la segunda expresión si no es necesario
def func():
    print("Se ejecutó func()")
    return True

False and func()  # No imprime nada (cortocircuito)
True or func()    # No imprime nada (cortocircuito)

# Pertenencia
lista = [1, 2, 3]
print(2 in lista)         # True
print(5 not in lista)     # True

# Identidad (mismo objeto en memoria)
a = [1, 2, 3]
b = [1, 2, 3]
c = a
print(a == b)    # True (mismo valor)
print(a is b)    # False (objetos diferentes)
print(a is c)    # True (mismo objeto)

# Encadenamiento de comparaciones
edad = 25
print(18 <= edad < 65)   # True (equivale a: 18 <= edad and edad < 65)

# Operadores sobre secuencias
lista1 = [1, 2]
lista2 = [3, 4]
print(lista1 + lista2)   # [1, 2, 3, 4] (concatenación)
print([0] * 3)           # [0, 0, 0] (repetición)
print("Python"[1:4])    # "yth" (slicing)
```

## 7. Control de flujo
- Selección: `if` / `elif` / `else`.
- Iteración:
  - `for` itera sobre elementos de un iterable.
  - `while` repite mientras la condición sea verdadera.
- Interrupciones: `break` (sale del bucle), `continue` (siguiente iteración), `pass` (no-op).
- Clausula `else` en bucles: se ejecuta si el bucle termina sin `break`.

```python
# Selección con if/elif/else
edad = 18
if edad < 13:
    categoria = "niño"
elif edad < 18:
    categoria = "adolescente"
elif edad < 65:
    categoria = "adulto"
else:
    categoria = "adulto mayor"
print(f"Categoría: {categoria}")

# Bucle for con iterable
frutas = ["manzana", "banana", "naranja"]
for fruta in frutas:
    print(f"Me gusta la {fruta}")

# Bucle for con range
for i in range(3):
    print(f"Iteración {i}")

# Bucle while
contador = 0
while contador < 3:
    print(f"Contador: {contador}")
    contador += 1

# Break y continue
for i in range(10):
    if i == 3:
        continue  # Salta esta iteración
    if i == 7:
        break     # Sale del bucle
    print(i)      # Imprime: 0, 1, 2, 4, 5, 6

# Clausula else en bucles
for i in range(3):
    print(i)
else:
    print("Bucle terminado normalmente")  # Se ejecuta

# Con break, else no se ejecuta
for i in range(5):
    if i == 2:
        break
    print(i)
else:
    print("Esto NO se imprime")  # No se ejecuta por el break

# Pass como placeholder
def funcion_pendiente():
    pass  # TODO: implementar después

class ClasePendiente:
    pass  # Placeholder para la implementación
```

## 8. Funciones (definición y semántica)
- Definición: `def nombre(parámetros):` con bloque indentado.
- Docstrings: la primera cadena del cuerpo documenta la función.
- Parámetros:
  - Posicionales y por palabra clave (keyword-only con `*`).
  - Valores por defecto evaluados en tiempo de definición (¡cuidado con mutables!).
  - Variádicos: `*args` (tupla de posicionales), `**kwargs` (dict de nombrados).
- Retorno: `return` (si se omite, devuelve `None`).
- Funciones son ciudadanos de primera clase: se pueden pasar como valores.

```python
# Función básica con docstring
def saludar(nombre):
    """Saluda a una persona por su nombre."""
    return f"Hola, {nombre}!"

# Parámetros con valores por defecto
def crear_perfil(nombre, edad=18, activo=True):
    return {"nombre": nombre, "edad": edad, "activo": activo}

# Parámetros keyword-only (después de *)
def configurar(host, puerto, *, debug=False, timeout=30):
    print(f"Conectando a {host}:{puerto}")
    if debug:
        print("Modo debug activado")

# Parámetros variádicos
def sumar(*numeros):
    """Suma una cantidad variable de números."""
    return sum(numeros)

def crear_usuario(nombre, **propiedades):
    """Crea un usuario con propiedades adicionales."""
    usuario = {"nombre": nombre}
    usuario.update(propiedades)
    return usuario

# Combinando tipos de parámetros
def funcion_completa(a, b, c=10, *args, d, e=20, **kwargs):
    print(f"a={a}, b={b}, c={c}")
    print(f"args={args}")
    print(f"d={d}, e={e}")
    print(f"kwargs={kwargs}")

# Funciones como ciudadanos de primera clase
def aplicar_operacion(func, x, y):
    """Aplica una función a dos argumentos."""
    return func(x, y)

def multiplicar(a, b):
    return a * b

# Uso de funciones como valores
resultado = aplicar_operacion(multiplicar, 3, 4)  # 12

# Funciones lambda (anónimas)
cuadrado = lambda x: x ** 2
numeros = [1, 2, 3, 4, 5]
cuadrados = list(map(cuadrado, numeros))  # [1, 4, 9, 16, 25]

# Función que no retorna explícitamente (retorna None)
def imprimir_info(dato):
    print(f"Información: {dato}")
    # return None (implícito)

# CUIDADO: valores por defecto mutables
def agregar_elemento(elemento, lista=[]):  # ¡MALO!
    lista.append(elemento)
    return lista

# CORRECTO: usar None como marcador
def agregar_elemento_correcto(elemento, lista=None):
    if lista is None:
        lista = []
    lista.append(elemento)
    return lista
```

## 9. Ámbito de nombres (LEGB)
- Resolución de nombres: Local → Enclosing (no local) → Global (módulo) → Builtins.
- `global` y `nonlocal` permiten enlazar a nombres de ámbitos externos específicos.

```python
# Variables globales (módulo)
contador_global = 0

def demostrar_legb():
    # Local: variable dentro de la función
    local_var = "soy local"

    def funcion_anidada():
        # Enclosing: variable de la función contenedora
        enclosing_var = "soy enclosing"

        def funcion_mas_anidada():
            # Local de esta función
            inner_local = "soy inner local"

            # Búsqueda LEGB
            print(inner_local)      # Local
            print(enclosing_var)    # Enclosing
            print(local_var)        # Enclosing (de la función padre)
            print(contador_global)  # Global
            print(len)              # Builtin

        funcion_mas_anidada()

    funcion_anidada()

# Modificación de variables globales
def incrementar_contador():
    global contador_global
    contador_global += 1
    return contador_global

# Uso de nonlocal
def crear_contador():
    cuenta = 0

    def incrementar():
        nonlocal cuenta  # Modifica la variable de la función contenedora
        cuenta += 1
        return cuenta

    def obtener():
        return cuenta  # Solo lectura, no necesita nonlocal

    return incrementar, obtener

# Ejemplo de uso de nonlocal
inc, get = crear_contador()
print(get())    # 0
print(inc())    # 1
print(inc())    # 2
print(get())    # 2

# Ejemplo de shadowing (sombreado)
x = "global"

def funcion_con_shadow():
    x = "local"  # Sombrea la variable global
    print(f"Dentro de la función: {x}")

funcion_con_shadow()  # "local"
print(f"Fuera de la función: {x}")  # "global"

# Sin modificador, se crea variable local
def funcion_shadow_problema():
    print(x)  # ¡Error! Se intenta leer x local antes de definirla
    x = "local"  # Python detecta la asignación y asume x es local

# Solución con global
def funcion_sin_shadow():
    global x
    print(x)  # Lee la variable global
    x = "modificada"  # Modifica la variable global
```

## 10. Módulos y paquetes
- Módulo: archivo `.py` que define un espacio de nombres propio.
- Paquete: carpeta con `__init__.py` que agrupa módulos y subpaquetes.
- Importación: crea una única instancia del módulo por proceso; se cachea en `sys.modules`.
- Atributo `__name__`: nombre cualificado del módulo; sirve para detectar contexto de importación.

```python
# === archivo: mi_modulo.py ===
"""Un módulo de ejemplo."""

# Variable del módulo
PI = 3.14159

# Función del módulo
def calcular_area_circulo(radio):
    """Calcula el área de un círculo."""
    return PI * radio ** 2

# Clase del módulo
class Calculadora:
    def sumar(self, a, b):
        return a + b

# Código que se ejecuta solo si el módulo se ejecuta directamente
if __name__ == "__main__":
    print("Ejecutando mi_modulo.py directamente")
    print(f"Área de círculo con radio 5: {calcular_area_circulo(5)}")

# === archivo principal ===
# Diferentes formas de importar

# Importar todo el módulo
import mi_modulo
area = mi_modulo.calcular_area_circulo(3)
calc = mi_modulo.Calculadora()

# Importar elementos específicos
from mi_modulo import PI, calcular_area_circulo
area = calcular_area_circulo(3)  # No necesita prefijo

# Importar con alias
import mi_modulo as mm
from mi_modulo import calcular_area_circulo as area_circulo
area = area_circulo(3)

# Importar todo (no recomendado)
from mi_modulo import *

# Importar módulos de la biblioteca estándar
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Estructura de paquete:
# mi_paquete/
#     __init__.py
#     modulo1.py
#     modulo2.py
#     sub_paquete/
#         __init__.py
#         modulo3.py

# Importar desde paquetes
from mi_paquete import modulo1
from mi_paquete.sub_paquete import modulo3
import mi_paquete.modulo2 as mod2

# === archivo: __init__.py en mi_paquete ===
"""Inicialización del paquete."""

# Importar y exponer elementos del paquete
from .modulo1 import funcion_util
from .modulo2 import ClaseImportante

# Define qué se importa con "from mi_paquete import *"
__all__ = ['funcion_util', 'ClaseImportante']

# Código de inicialización del paquete
print("Inicializando mi_paquete")

# Importación relativa (dentro del paquete)
# from . import modulo_hermano
# from ..otro_paquete import otro_modulo
# from .sub_paquete.modulo3 import funcion

# Atributo __name__ examples
print(f"Nombre del módulo: {__name__}")
# Si se ejecuta directamente: "__main__"
# Si se importa: "nombre_del_modulo"

# Verificar si un módulo está cargado
import sys
if 'mi_modulo' in sys.modules:
    print("mi_modulo ya está cargado")
```

## 11. Manejo de errores (excepciones)
- Excepciones son objetos que interrumpen el flujo normal; se capturan con `try/except`.
- Clausulas:
  - `try/except` (captura), `try/except/else` (código si no hubo excepción), `try/finally` (limpieza), o combinadas.
- Propagación: si no se captura, la excepción burbujea hasta abortar el programa.
- Uso de `raise` para señalar condiciones inválidas o violaciones de contrato.

```python
# Manejo básico de excepciones
def dividir(a, b):
    try:
        resultado = a / b
        return resultado
    except ZeroDivisionError:
        print("Error: División por cero")
        return None
    except TypeError:
        print("Error: Tipos de datos incorrectos")
        return None

# Captura múltiples excepciones
def procesar_numero(valor):
    try:
        numero = int(valor)
        resultado = 100 / numero
        return resultado
    except (ValueError, TypeError):
        print("Error: Valor no convertible a número")
    except ZeroDivisionError:
        print("Error: División por cero")
    except Exception as e:
        print(f"Error inesperado: {e}")

# Try/except/else/finally
def leer_archivo(nombre):
    archivo = None
    try:
        archivo = open(nombre, 'r')
        contenido = archivo.read()
    except FileNotFoundError:
        print(f"Archivo {nombre} no encontrado")
        return None
    except IOError:
        print(f"Error al leer el archivo {nombre}")
        return None
    else:
        # Se ejecuta solo si NO hay excepción
        print("Archivo leído exitosamente")
        return contenido
    finally:
        # Se ejecuta SIEMPRE
        if archivo:
            archivo.close()
            print("Archivo cerrado")

# Lanzar excepciones con raise
def validar_edad(edad):
    if not isinstance(edad, int):
        raise TypeError("La edad debe ser un entero")
    if edad < 0:
        raise ValueError("La edad no puede ser negativa")
    if edad > 150:
        raise ValueError("La edad no puede ser mayor a 150")
    return edad

# Excepciones personalizadas
class EdadInvalidaError(Exception):
    """Excepción personalizada para edades inválidas."""
    def __init__(self, edad, mensaje="Edad inválida"):
        self.edad = edad
        self.mensaje = mensaje
        super().__init__(self.mensaje)

    def __str__(self):
        return f"{self.mensaje}: {self.edad}"

def validar_edad_custom(edad):
    if edad < 0 or edad > 150:
        raise EdadInvalidaError(edad, "La edad debe estar entre 0 y 150")
    return edad

# Uso de excepciones personalizadas
try:
    validar_edad_custom(-5)
except EdadInvalidaError as e:
    print(f"Error capturado: {e}")

# Re-lanzar excepciones
def funcion_wrapper():
    try:
        resultado = dividir(10, 0)
    except ZeroDivisionError:
        print("Manejando error en wrapper")
        raise  # Re-lanza la misma excepción

# Context manager implícito con try/finally
def gestionar_recurso():
    recurso = None
    try:
        recurso = "abrir_recurso()"
        # Usar recurso
        print("Usando recurso")
        # Simular error
        raise ValueError("Algo salió mal")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if recurso:
            print("Cerrando recurso")

# Jerarquía de excepciones
"""
BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- GeneratorExit
 +-- Exception
      +-- StopIteration
      +-- ArithmeticError
      |    +-- ZeroDivisionError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- ValueError
      +-- TypeError
      +-- ...
"""

# Buenas prácticas
def funcion_robusta(datos):
    if not datos:
        raise ValueError("Los datos no pueden estar vacíos")

    try:
        # Procesamiento específico
        return procesar_datos(datos)
    except KeyError as e:
        # Error específico y útil
        raise ValueError(f"Falta la clave requerida: {e}") from e
    except Exception:
        # Log del error y re-lanzar
        print("Error inesperado en procesamiento")
        raise
```

## 12. Objetos y clases (nociones básicas)
- Clase define un tipo; instancia es un objeto de esa clase.
- Métodos de instancia reciben `self` como primer parámetro por convención.
- Atributos pueden definirse en la clase (compartidos) o en la instancia.
- Métodos estáticos y de clase: `@staticmethod` y `@classmethod`.
- Modelo de herencia simple; resolución de atributos vía MRO (orden de resolución de métodos).

```python
# Definición básica de clase
class Persona:
    """Clase que representa a una persona."""

    # Atributo de clase (compartido por todas las instancias)
    especie = "Homo sapiens"

    def __init__(self, nombre, edad):
        """Constructor de la clase."""
        # Atributos de instancia
        self.nombre = nombre
        self.edad = edad
        self.activo = True

    def saludar(self):
        """Método de instancia."""
        return f"Hola, soy {self.nombre}"

    def cumplir_años(self):
        """Modifica el estado de la instancia."""
        self.edad += 1
        return self.edad

    def __str__(self):
        """Representación string del objeto."""
        return f"Persona({self.nombre}, {self.edad} años)"

    def __repr__(self):
        """Representación técnica del objeto."""
        return f"Persona('{self.nombre}', {self.edad})"

# Crear instancias
persona1 = Persona("Ana", 25)
persona2 = Persona("Carlos", 30)

# Usar métodos y atributos
print(persona1.saludar())        # "Hola, soy Ana"
print(persona1.nombre)           # "Ana"
print(persona1.especie)          # "Homo sapiens"
persona1.cumplir_años()
print(persona1.edad)             # 26

# Métodos estáticos y de clase
class Calculadora:
    pi = 3.14159

    def __init__(self, precision=2):
        self.precision = precision

    def redondear(self, numero):
        """Método de instancia."""
        return round(numero, self.precision)

    @staticmethod
    def sumar(a, b):
        """Método estático (no accede a self ni cls)."""
        return a + b

    @classmethod
    def crear_cientifica(cls):
        """Método de clase (recibe cls, la clase)."""
        return cls(precision=10)  # Crea instancia con precisión alta

    @classmethod
    def obtener_pi(cls):
        """Accede a atributos de clase."""
        return cls.pi

# Uso de métodos estáticos y de clase
calc_normal = Calculadora()
calc_cientifica = Calculadora.crear_cientifica()

print(Calculadora.sumar(5, 3))      # 8 (no necesita instancia)
print(Calculadora.obtener_pi())     # 3.14159
print(calc_cientifica.precision)    # 10

# Herencia
class Empleado(Persona):
    """Empleado hereda de Persona."""

    def __init__(self, nombre, edad, salario):
        super().__init__(nombre, edad)  # Llama al constructor padre
        self.salario = salario

    def saludar(self):
        """Override del método padre."""
        return f"Hola, soy {self.nombre} y trabajo aquí"

    def trabajar(self):
        """Método específico de Empleado."""
        return f"{self.nombre} está trabajando"

# Herencia múltiple (ejemplo conceptual)
class Volador:
    def volar(self):
        return "Volando..."

class Nadador:
    def nadar(self):
        return "Nadando..."

class Pato(Volador, Nadador):
    def __init__(self, nombre):
        self.nombre = nombre

    def hacer_ruido(self):
        return "Cuac!"

# Uso de herencia
empleado = Empleado("Luis", 28, 50000)
print(empleado.saludar())           # Método override
print(empleado.cumplir_años())      # Método heredado
print(empleado.trabajar())          # Método específico

pato = Pato("Donald")
print(pato.volar())                 # Método de Volador
print(pato.nadar())                 # Método de Nadador

# MRO (Method Resolution Order)
print(Pato.__mro__)                 # Orden de resolución de métodos

# Atributos privados (convención con _)
class CuentaBancaria:
    def __init__(self, saldo_inicial):
        self.titular = "Propietario"
        self._saldo = saldo_inicial      # Protegido (convención)
        self.__numero = "123456789"      # Privado (name mangling)

    def obtener_saldo(self):
        return self._saldo

    def _metodo_interno(self):
        return "Método interno"

    def __metodo_privado(self):
        return "Método privado"

# Propiedades con decoradores
class Circulo:
    def __init__(self, radio):
        self._radio = radio

    @property
    def radio(self):
        """Getter para radio."""
        return self._radio

    @radio.setter
    def radio(self, valor):
        """Setter para radio."""
        if valor <= 0:
            raise ValueError("El radio debe ser positivo")
        self._radio = valor

    @property
    def area(self):
        """Propiedad calculada."""
        return 3.14159 * self._radio ** 2

# Uso de propiedades
circulo = Circulo(5)
print(circulo.radio)        # 5
print(circulo.area)         # 78.53975
circulo.radio = 10          # Usa el setter
print(circulo.area)         # 314.159
```

## 13. Anotaciones de tipo (type hints)
- Sintaxis PEP 484: opcionales, no afectan la ejecución en CPython.
- Útiles para herramientas (editores, linters, analizadores estáticos).
- Ejemplos conceptuales: `x: int`, `def f(a: str) -> bool: ...`, `Optional[T]`, `Union[...]`, `Iterable[T]`.

```python
# Importar tipos del módulo typing (Python < 3.9)
from typing import List, Dict, Optional, Union, Callable, Tuple, Any, Iterable

# Anotaciones básicas de variables
nombre: str = "Juan"
edad: int = 25
activo: bool = True
precio: float = 19.99

# Función con anotaciones de parámetros y retorno
def saludar(nombre: str, formal: bool = False) -> str:
    """Función con type hints."""
    if formal:
        return f"Buenos días, {nombre}"
    else:
        return f"Hola {nombre}!"

# Tipos de colecciones
numeros: List[int] = [1, 2, 3, 4, 5]
personas: Dict[str, int] = {"Ana": 25, "Carlos": 30}
coordenadas: Tuple[float, float] = (10.5, 20.3)

# Optional (puede ser None)
def buscar_usuario(id: int) -> Optional[str]:
    """Retorna el nombre del usuario o None si no existe."""
    usuarios = {1: "Ana", 2: "Carlos"}
    return usuarios.get(id)

# Union (múltiples tipos posibles)
def procesar_id(id: Union[int, str]) -> str:
    """Acepta ID como entero o string."""
    return f"ID procesado: {id}"

# Funciones como parámetros
def aplicar_operacion(x: int, y: int, operacion: Callable[[int, int], int]) -> int:
    """Aplica una función a dos números."""
    return operacion(x, y)

def sumar(a: int, b: int) -> int:
    return a + b

resultado = aplicar_operacion(5, 3, sumar)

# Tipos genéricos con TypeVar (Python >= 3.5)
from typing import TypeVar

T = TypeVar('T')

def primer_elemento(lista: List[T]) -> T:
    """Retorna el primer elemento de la lista."""
    return lista[0]

# Uso con diferentes tipos
primer_numero: int = primer_elemento([1, 2, 3])
primera_palabra: str = primer_elemento(["hola", "mundo"])

# Clases con anotaciones
class Persona:
    def __init__(self, nombre: str, edad: int) -> None:
        self.nombre: str = nombre
        self.edad: int = edad
        self.amigos: List[str] = []

    def añadir_amigo(self, amigo: str) -> None:
        self.amigos.append(amigo)

    def obtener_info(self) -> Dict[str, Union[str, int]]:
        return {"nombre": self.nombre, "edad": self.edad}

# Anotaciones avanzadas
def procesar_datos(
    datos: Iterable[str],
    filtros: Optional[List[str]] = None,
    callback: Optional[Callable[[str], None]] = None
) -> Dict[str, int]:
    """Procesa datos con filtros opcionales y callback."""
    resultado = {}
    for dato in datos:
        if filtros is None or dato in filtros:
            resultado[dato] = len(dato)
            if callback:
                callback(dato)
    return resultado

# Python 3.9+ sintaxis simplificada (sin imports)
# def nueva_funcion(items: list[str]) -> dict[str, int]:
#     return {item: len(item) for item in items}

# Type aliases para claridad
PersonaDict = Dict[str, Union[str, int]]
ListaNumeros = List[int]

def crear_persona_dict(nombre: str, edad: int) -> PersonaDict:
    return {"nombre": nombre, "edad": edad}

# Anotaciones para métodos especiales
class Contador:
    def __init__(self, inicial: int = 0) -> None:
        self.valor: int = inicial

    def __add__(self, otro: int) -> 'Contador':
        return Contador(self.valor + otro)

    def __str__(self) -> str:
        return f"Contador({self.valor})"

# Any para cuando no conocemos el tipo
def procesar_cualquier_cosa(dato: Any) -> str:
    """Acepta cualquier tipo de dato."""
    return str(dato)

# Anotaciones en variables de instancia (PEP 526)
class ConfiguracionApp:
    debug: bool
    host: str
    puerto: int

    def __init__(self) -> None:
        self.debug = True
        self.host = "localhost"
        self.puerto = 8000

# Literal types (Python 3.8+)
from typing import Literal

def configurar_modo(modo: Literal["desarrollo", "produccion"]) -> None:
    print(f"Configurando en modo: {modo}")

# Final (Python 3.8+)
from typing import Final

API_VERSION: Final[str] = "v1.0"  # Constante

# Protocol para duck typing (Python 3.8+)
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def renderizar(obj: Drawable) -> None:
    obj.draw()

class Circulo:
    def draw(self) -> None:
        print("Dibujando círculo")

# Funciona sin herencia explícita
renderizar(Circulo())
```

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