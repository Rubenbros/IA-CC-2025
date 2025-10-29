# Exportación de datos a ficheros CSV

Los ficheros CSV (Comma-Separated Values) son uno de los formatos más comunes para intercambiar datos tabulares. En este documento aprenderás a exportar datos a CSV tanto usando pandas como Python estándar.

## ¿Qué es un fichero CSV?

Un fichero CSV es un archivo de texto plano que almacena datos en formato tabular donde:
- Cada línea representa una fila
- Los valores están separados por comas (u otro delimitador)
- La primera línea usualmente contiene los nombres de las columnas

Ejemplo de contenido CSV:
```csv
nombre,edad,ciudad
Ana,25,Madrid
Carlos,30,Barcelona
María,28,Valencia
```

## Exportar con pandas

pandas proporciona el método `to_csv()` que es muy potente y flexible para exportar DataFrames a ficheros CSV.

### Sintaxis básica

```python
import pandas as pd

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'nombre': ['Ana', 'Carlos', 'María'],
    'edad': [25, 30, 28],
    'ciudad': ['Madrid', 'Barcelona', 'Valencia']
})

# Exportar a CSV
df.to_csv('datos.csv')
```

### Parámetros importantes de to_csv()

#### 1. **index** (bool)
Controla si se exporta el índice del DataFrame:

```python
# Sin índice
df.to_csv('sin_indice.csv', index=False)

# Con índice (por defecto)
df.to_csv('con_indice.csv', index=True)
```

#### 2. **sep** (str)
Define el separador de campos:

```python
# Punto y coma como separador
df.to_csv('datos_semicolon.csv', sep=';')

# Tabulador como separador
df.to_csv('datos_tab.csv', sep='\t')
```

#### 3. **encoding** (str)
Especifica la codificación del archivo:

```python
# UTF-8 (recomendado para caracteres especiales)
df.to_csv('datos_utf8.csv', encoding='utf-8')

# Latin-1 (compatible con Excel en español)
df.to_csv('datos_latin1.csv', encoding='latin-1')
```

#### 4. **columns** (list)
Exporta solo columnas específicas:

```python
# Exportar solo nombre y edad
df.to_csv('columnas_seleccionadas.csv', columns=['nombre', 'edad'])
```

#### 5. **header** (bool o list)
Controla los nombres de columnas:

```python
# Sin encabezados
df.to_csv('sin_headers.csv', header=False)

# Con nombres personalizados
df.to_csv('headers_custom.csv', header=['Name', 'Age', 'City'])
```

#### 6. **mode** y **header** para añadir datos
Añadir datos a un archivo existente:

```python
# Primera escritura
df1.to_csv('acumulativo.csv', index=False)

# Añadir más datos
df2.to_csv('acumulativo.csv', mode='a', header=False, index=False)
```

### Manejo de valores nulos

```python
import numpy as np

df_con_nulos = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4, 5, np.nan]
})

# Representar nulos como cadena vacía
df_con_nulos.to_csv('nulos.csv', na_rep='')

# Representar nulos como 'N/A'
df_con_nulos.to_csv('nulos_na.csv', na_rep='N/A')
```

## Exportar con Python estándar

Python incluye el módulo `csv` en su biblioteca estándar, que permite trabajar con ficheros CSV sin necesidad de pandas.

### Usando csv.writer

```python
import csv

# Datos a exportar
datos = [
    ['nombre', 'edad', 'ciudad'],
    ['Ana', 25, 'Madrid'],
    ['Carlos', 30, 'Barcelona'],
    ['María', 28, 'Valencia']
]

# Escribir CSV
with open('datos_python.csv', 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerows(datos)
```

### Usando csv.DictWriter

Útil cuando los datos están en formato de diccionarios:

```python
import csv

# Datos como lista de diccionarios
personas = [
    {'nombre': 'Ana', 'edad': 25, 'ciudad': 'Madrid'},
    {'nombre': 'Carlos', 'edad': 30, 'ciudad': 'Barcelona'},
    {'nombre': 'María', 'edad': 28, 'ciudad': 'Valencia'}
]

# Definir nombres de columnas
columnas = ['nombre', 'edad', 'ciudad']

# Escribir CSV
with open('datos_dict.csv', 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.DictWriter(archivo, fieldnames=columnas)

    # Escribir encabezados
    escritor.writeheader()

    # Escribir filas
    escritor.writerows(personas)
```

### Personalizar el formato CSV

```python
import csv

# Crear un dialecto personalizado
csv.register_dialect('mi_dialecto',
                    delimiter=';',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)

# Usar el dialecto personalizado
with open('custom.csv', 'w', newline='') as archivo:
    escritor = csv.writer(archivo, dialect='mi_dialecto')
    escritor.writerow(['campo1', 'campo2', 'campo3'])
    escritor.writerow(['valor1', 'valor2', 'valor3'])
```

### Escribir línea por línea

```python
import csv

with open('linea_por_linea.csv', 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)

    # Escribir encabezados
    escritor.writerow(['producto', 'precio', 'cantidad'])

    # Escribir datos línea por línea
    escritor.writerow(['Laptop', 999.99, 5])
    escritor.writerow(['Mouse', 25.50, 15])
    escritor.writerow(['Teclado', 75.00, 10])
```

## Comparación: pandas vs Python estándar

| Característica | pandas | Python estándar (csv) |
|---------------|--------|----------------------|
| **Instalación** | Requiere instalación | Incluido en Python |
| **Velocidad** | Más rápido para datos grandes | Más ligero para datos pequeños |
| **Funcionalidad** | Muchas opciones de exportación | Funcionalidad básica |
| **Manejo de tipos** | Automático | Manual |
| **Valores nulos** | Manejo automático | Requiere lógica adicional |
| **Memoria** | Mayor consumo | Menor consumo |

## Casos de uso recomendados

### Usa pandas cuando:
- Trabajas con DataFrames existentes
- Necesitas opciones avanzadas de exportación
- Manejas grandes volúmenes de datos
- Requieres conversión automática de tipos
- Trabajas con fechas y valores nulos

### Usa Python estándar cuando:
- No quieres dependencias externas
- Trabajas con datos simples
- Necesitas control total sobre el formato
- La aplicación debe ser ligera
- Procesas datos línea por línea

## Ejemplos prácticos

Ver los archivos de ejemplo en `src/clase04_pandas/`:
- `exportar_csv_pandas.py`: Ejemplos completos con pandas
- `exportar_csv_python.py`: Ejemplos con Python estándar

## Ejercicios propuestos

1. Crea un DataFrame con datos de productos (nombre, precio, stock) y expórtalo a CSV sin índice
2. Lee un archivo CSV existente y expórtalo con un delimitador diferente
3. Implementa una función que exporte datos a CSV usando Python estándar con manejo de errores
4. Crea un script que acumule datos en un CSV existente sin duplicar encabezados