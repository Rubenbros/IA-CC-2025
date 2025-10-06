# Clase 03 — Estructuras de datos y objetos en Python

Buenas a todos. Hoy vamos a trabajar con las estructuras de datos fundamentales de Python y a profundizar en programación orientada a objetos. Sé que algunos habéis tenido dificultades con programación en primero, así que vamos a ir paso a paso y con muchos ejemplos prácticos.

## Objetivos de la clase 03
- Dominar las estructuras de datos básicas: listas, tuplas, diccionarios y conjuntos
- Entender cuándo usar cada una y sus diferencias clave
- Profundizar en POO: clases, herencia, métodos especiales
- Aplicar todo en ejercicios prácticos que podréis reutilizar

## Contenido y código de esta clase
- Código principal: `src/clase03_estructuras_datos_objetos/`
- Requisitos: ninguno especial, Python estándar

## Contenido teórico

### 1. Listas (list)
Las listas son **mutables** (se pueden modificar) y **ordenadas**. Son como los ArrayList de Java, pero mucho más flexibles.

**Características:**
- Se definen con corchetes `[]`
- Pueden contener elementos de diferentes tipos
- Permiten elementos duplicados
- Indexación empieza en 0 (como en Java)
- Soportan índices negativos: `-1` es el último elemento

**Operaciones principales:**
- `append(x)`: añade al final
- `insert(i, x)`: inserta en posición i
- `remove(x)`: elimina la primera ocurrencia
- `pop()`: elimina y devuelve el último (o el índice indicado)
- `sort()`: ordena en el sitio
- `len(lista)`: longitud
- Slicing: `lista[inicio:fin:paso]`

### 2. Tuplas (tuple)
Las tuplas son **inmutables** (no se pueden modificar después de crearlas) y **ordenadas**.

**Características:**
- Se definen con paréntesis `()` o sin ellos
- Más rápidas que las listas (optimización)
- Útiles para datos que no deben cambiar
- Se pueden usar como claves de diccionario (las listas no)

**¿Cuándo usar tuplas en vez de listas?**
- Coordenadas: `(x, y, z)`
- Datos que no deben cambiar: `fecha = (2025, 1, 15)`
- Retornar múltiples valores desde una función
- Como claves de diccionarios

### 3. Diccionarios (dict)
Los diccionarios son colecciones de pares **clave-valor**, **mutables** y **sin orden garantizado** (aunque desde Python 3.7+ mantienen el orden de inserción).

**Características:**
- Se definen con llaves `{clave: valor}`
- Las claves deben ser inmutables (strings, números, tuplas)
- Muy rápidos para búsquedas (O(1) en promedio)
- Equivalente a HashMap en Java

**Operaciones principales:**
- `dict[clave]`: acceso (lanza KeyError si no existe)
- `dict.get(clave, default)`: acceso seguro con valor por defecto
- `dict[clave] = valor`: asignación/modificación
- `del dict[clave]`: eliminar
- `clave in dict`: comprobar existencia
- `dict.keys()`, `dict.values()`, `dict.items()`: vistas

### 4. Conjuntos (set)
Los conjuntos son colecciones **sin orden** y **sin duplicados**, **mutables**.

**Características:**
- Se definen con llaves `{elem1, elem2}` o `set()`
- No permiten elementos duplicados (automáticamente los elimina)
- Ideales para operaciones matemáticas de conjuntos
- Muy rápidos para comprobar pertenencia

**Operaciones principales:**
- `add(x)`: añadir elemento
- `remove(x)`: eliminar (error si no existe)
- `discard(x)`: eliminar (sin error si no existe)
- Operaciones de conjuntos: `union()`, `intersection()`, `difference()`, `symmetric_difference()`
- Operadores: `|` (unión), `&` (intersección), `-` (diferencia)

### 5. Comparación rápida

| Estructura | Mutable | Ordenada | Duplicados | Acceso | Uso típico |
|------------|---------|----------|------------|--------|------------|
| **Lista** | Sí | Sí | Sí | Por índice | Colecciones ordenadas modificables |
| **Tupla** | No | Sí | Sí | Por índice | Datos inmutables, coordenadas |
| **Diccionario** | Sí | Sí* | Claves únicas | Por clave | Mapeos clave-valor |
| **Conjunto** | Sí | No | No | Por valor | Elementos únicos, operaciones de conjuntos |

*Desde Python 3.7+ mantienen orden de inserción, pero no es su característica principal.

## Programación Orientada a Objetos (POO) en Python

### Conceptos clave que debéis dominar

**Clases y objetos:**
- Clase: plantilla o molde (como en Java)
- Objeto: instancia de una clase
- `__init__`: constructor (equivalente al constructor en Java)
- `self`: referencia al objeto actual (equivalente a `this` en Java)

**Diferencias importantes con Java:**
- En Python, `self` debe ser explícito como primer parámetro
- No hay modificadores de acceso reales (`private`, `protected`, `public`)
- Convención: `_atributo` (un guion bajo) = "protegido" por convención
- Convención: `__atributo` (dos guiones bajos) = name mangling (casi privado)

**Métodos especiales (dunder methods):**
- `__init__(self, ...)`: constructor
- `__str__(self)`: representación legible para humanos (como `toString()` en Java)
- `__repr__(self)`: representación técnica del objeto
- `__eq__(self, other)`: comparación de igualdad (`==`)
- `__lt__(self, other)`: menor que (`<`)
- Muchos más: `__len__`, `__add__`, `__getitem__`, etc.

**Herencia:**
- Sintaxis: `class Hija(Padre):`
- `super()`: llama a métodos de la clase padre
- Python soporta herencia múltiple (con cuidado)

**Propiedades:**
- `@property`: convertir un método en atributo de solo lectura
- `@atributo.setter`: definir cómo se asigna el valor
- Alternativa a los getters/setters de Java, más pythónico

### Diferencias con Java que debéis tener claras

```
Java                          Python
----                          ------
public/private/protected      Convenciones (_var, __var)
this                          self (explícito)
new MiClase()                 MiClase()
toString()                    __str__()
equals()                      __eq__()
extends                       class Hija(Padre):
interface                     ABC (Abstract Base Classes)
static                        @staticmethod o @classmethod
final                         No existe, usar convenciones
```

## Cómo ejecutar los ejemplos

Cada script se puede ejecutar de forma independiente:

```bash
# Ejemplos de estructuras de datos
python -m src.clase03_estructuras_datos_objetos.listas
python -m src.clase03_estructuras_datos_objetos.tuplas
python -m src.clase03_estructuras_datos_objetos.diccionarios
python -m src.clase03_estructuras_datos_objetos.conjuntos

# Ejemplos de POO
python -m src.clase03_estructuras_datos_objetos.clases_basicas
python -m src.clase03_estructuras_datos_objetos.herencia
python -m src.clase03_estructuras_datos_objetos.metodos_especiales
python -m src.clase03_estructuras_datos_objetos.propiedades

# Ejercicio integrador
python -m src.clase03_estructuras_datos_objetos.ejercicio_biblioteca
```

## Consejos para los que habéis tenido dificultades

1. **Practicad con listas antes de pasar a otras estructuras.** Son las más versátiles.

2. **No os agobiéis con todas las estructuras a la vez.** Empezad por listas y diccionarios, que son las más usadas.

3. **Dibujad en papel cómo se organizan los datos.** Sobre todo con diccionarios anidados.

4. **Usad `type()` y `print()` constantemente** para ver qué tenéis en cada variable.

5. **Los errores son normales.** `KeyError`, `IndexError`, `TypeError` son vuestros amigos: os dicen exactamente qué está mal.

6. **Para POO: empezad con clases simples.** Una clase Persona con nombre y edad. Cuando funcione, añadid más.

7. **Preguntad.** Si algo no queda claro, levantad la mano. Esto no es trivial y es normal tener dudas.

## Recursos adicionales

- Documentación oficial de Python sobre estructuras: https://docs.python.org/3/tutorial/datastructures.html
- Python Tutor (visualizar ejecución paso a paso): https://pythontutor.com/
- Real Python - Lists and Tuples: https://realpython.com/python-lists-tuples/
- Real Python - Dictionaries: https://realpython.com/python-dicts/

