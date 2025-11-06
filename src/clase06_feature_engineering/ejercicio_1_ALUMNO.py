"""
EJERCICIO 1: FRECUENCIAS Y AGREGACIONES BASICAS
================================================

OBJETIVO: Aprender a crear features de agregación a partir de datos de ventas.

INSTRUCCIONES:
1. Lee los comentarios TODO
2. Completa el código donde dice "# TU CÓDIGO AQUÍ"
3. Ejecuta el script para ver si tus features son correctas
4. Las funciones de verificación te dirán si está bien o mal

NO NECESITAS SABER:
- Matplotlib (nosotros lo manejamos)
- Pandas avanzado (te damos funciones auxiliares)

SOLO NECESITAS:
- Operaciones básicas: suma, resta, multiplicación, división
- Contar elementos
- Calcular promedios
"""

import pandas as pd

# =============================================================================
# FUNCIONES AUXILIARES (YA IMPLEMENTADAS - NO MODIFICAR)
# =============================================================================

def cargar_datos():
    """Carga el dataset de ventas"""
    df = pd.read_csv('ejercicio_1_ventas.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def verificar_feature(nombre_feature, valor_calculado, valor_esperado, tolerancia=0.01):
    """
    Verifica si la feature calculada es correcta

    Args:
        nombre_feature: Nombre de la feature
        valor_calculado: Tu resultado
        valor_esperado: Resultado correcto
        tolerancia: Margen de error aceptable
    """
    if isinstance(valor_calculado, (int, float)) and isinstance(valor_esperado, (int, float)):
        es_correcto = abs(valor_calculado - valor_esperado) < tolerancia
    else:
        es_correcto = valor_calculado == valor_esperado

    if es_correcto:
        print(f" {nombre_feature}: CORRECTO")
        print(f"   Tu resultado: {valor_calculado}")
    else:
        print(f" {nombre_feature}: INCORRECTO")
        print(f"   Tu resultado: {valor_calculado}")
        print(f"   Esperado: {valor_esperado}")
        print(f"    Pista: Revisa la fórmula o el cálculo")
    print()


def mostrar_primeras_filas(df, n=5):
    """Muestra las primeras filas del dataset"""
    print("Primeras filas del dataset:")
    print(df.head(n))
    print()


# =============================================================================
# EJERCICIO 1.1: FEATURE DERIVADA SIMPLE
# =============================================================================

def ejercicio_1_1_total_venta(df):
    """
    TAREA: Crear una feature 'total_venta' que sea cantidad * precio_unitario

    Ejemplo:
    - cantidad = 2, precio_unitario = 10.50
    - total_venta = 2 * 10.50 = 21.00
    """
    print("="*70)
    print("EJERCICIO 1.1: Feature Derivada - Total de Venta")
    print("="*70)

    print("\n TEORÍA:")
    print("Una feature DERIVADA se crea combinando otras features existentes.")
    print("Formula: total_venta = cantidad × precio_unitario")

    print("\n Datos disponibles:")
    print(df[['cantidad', 'precio_unitario']].head())

    # TODO 1.1: Calcula el total de venta para cada fila
    # Pista: Multiplica la columna 'cantidad' por 'precio_unitario'

    # TU CÓDIGO AQUÍ (descomenta y completa):
    df['total_venta'] = df['cantidad'] * df['precio_unitario']

    # Verificación automática
    valor_esperado_fila_0 = 2 * 899.99  # Primera fila
    if 'total_venta' in df.columns:
        verificar_feature(
            "Total venta (primera fila)",
            df.loc[0, 'total_venta'],
            valor_esperado_fila_0
        )
    else:
        print(" No has creado la columna 'total_venta'\n")

    return df


# =============================================================================
# EJERCICIO 1.2: AGREGACIÓN - SUMA
# =============================================================================

def ejercicio_1_2_cantidad_total_por_producto(df):
    """
    TAREA: Calcular cuántas unidades se vendieron de CADA producto en total

    Ejemplo:
    - laptop aparece 5 veces: cantidades [2, 1, 1, 2, 1] → total = 7
    """
    print("="*70)
    print("EJERCICIO 1.2: Agregación - Cantidad Total por Producto")
    print("="*70)

    print("\n TEORÍA:")
    print("AGREGAR = combinar múltiples filas en un solo valor")
    print("Queremos: Por cada producto, sumar todas sus cantidades")

    print("\n Ejemplo:")
    print("laptop aparece en:")
    print(df[df['producto'] == 'laptop'][['producto', 'cantidad']])
    print(f"Total laptops vendidas: {df[df['producto'] == 'laptop']['cantidad'].sum()}")

    # TODO 1.2: Agrupa por producto y suma las cantidades
    # Pista: Usa df.groupby('producto')['cantidad'].sum()

    # TU CÓDIGO AQUÍ:
    cantidad_por_producto = df.groupby('producto')['cantidad'].sum()

    # Verificación
    esperado_laptop = 12  # Valor correcto para laptop
    if 'laptop' in cantidad_por_producto.index:
        verificar_feature(
            "Total laptops vendidas",
            cantidad_por_producto['laptop'],
            esperado_laptop
        )

    print(" Tu resultado:")
    print(cantidad_por_producto)
    print()

    return cantidad_por_producto


# =============================================================================
# EJERCICIO 1.3: AGREGACIÓN - FRECUENCIA
# =============================================================================

def ejercicio_1_3_frecuencia_productos(df):
    """
    TAREA: Calcular qué porcentaje representa cada producto del total

    Ejemplo:
    - Total transacciones: 30
    - laptop aparece: 5 veces
    - Frecuencia: 5/30 = 0.167 = 16.7%
    """
    print("="*70)
    print("EJERCICIO 1.3: Frecuencia de Productos")
    print("="*70)

    print("\n TEORÍA:")
    print("FRECUENCIA = cuántas veces aparece / total")
    print("Se expresa como decimal (0.0 a 1.0) o porcentaje (0% a 100%)")

    print(f"\n Info:")
    print(f"Total de transacciones: {len(df)}")
    print(f"laptop aparece: {(df['producto'] == 'laptop').sum()} veces")

    # TODO 1.3: Calcula la frecuencia de cada producto
    # Pista: (número de apariciones) / (total de filas)

    # PASO 1: Cuenta cuántas veces aparece cada producto
    # TU CÓDIGO AQUÍ:
    conteo = df['producto'].value_counts()

    # PASO 2: Divide por el total para obtener frecuencia
    # TU CÓDIGO AQUÍ:
    frecuencia = conteo / len(df)

    # Verificación
    esperado_laptop_freq = 7 / 30  # laptop aparece 10 veces de 30 total
    if 'laptop' in frecuencia.index:
        verificar_feature(
            "Frecuencia de laptop",
            frecuencia['laptop'],
            esperado_laptop_freq,
            tolerancia=0.01
        )

    print(" Tu resultado (en %):")
    print((frecuencia * 100).round(2))
    print()

    return frecuencia


# =============================================================================
# EJERCICIO 1.4: AGREGACIÓN POR GRUPO - PROMEDIO
# =============================================================================

def ejercicio_1_4_precio_promedio_por_categoria(df):
    """
    TAREA: Calcular el precio promedio de cada categoría

    Ejemplo:
    - Categoría 'electronica': precios [899.99, 25.50, 75.00, ...]
    - Promedio: suma de todos / cantidad
    """
    print("="*70)
    print("EJERCICIO 1.4: Precio Promedio por Categoría")
    print("="*70)

    print("\n TEORÍA:")
    print("PROMEDIO = suma de valores / cantidad de valores")
    print("Agrupamos por categoría y calculamos el promedio de precio_unitario")

    print("\n Ejemplo - Categoría 'electronica':")
    print(df[df['categoria'] == 'electronica'][['producto', 'precio_unitario']].head())

    # TODO 1.4: Agrupa por categoría y calcula el promedio de precio
    # Pista: df.groupby('categoria')['precio_unitario'].mean()

    # TU CÓDIGO AQUÍ:
    # precio_promedio = df.groupby('_______')['precio_unitario']._____()

    # ️ BORRA ESTAS LÍNEAS cuando hagas tu código:
    precio_promedio = pd.Series([0, 0], index=['electronica', 'muebles'])

    # Verificación
    # Precio promedio real de electrónica
    esperado_electronica = df[df['categoria'] == 'electronica']['precio_unitario'].mean()
    if 'electronica' in precio_promedio.index:
        verificar_feature(
            "Precio promedio electrónica",
            precio_promedio['electronica'],
            esperado_electronica,
            tolerancia=1.0
        )

    print(" Tu resultado:")
    print(precio_promedio.round(2))
    print()

    return precio_promedio


# =============================================================================
# EJERCICIO 1.5: VENTANA DESLIZANTE - PROMEDIO MÓVIL (AVANZADO)
# =============================================================================

def ejercicio_1_5_promedio_movil(df):
    """
    TAREA: Calcular el promedio de las últimas 3 ventas

    Ejemplo:
    - Ventas diarias: [1000, 1500, 1200, 1800, ...]
    - Promedio móvil día 3: (1000 + 1500 + 1200) / 3 = 1233
    - Promedio móvil día 4: (1500 + 1200 + 1800) / 3 = 1500
    """
    print("="*70)
    print("EJERCICIO 1.5: Promedio Móvil (Ventana de 3 días)")
    print("="*70)

    print("\n TEORÍA:")
    print("PROMEDIO MÓVIL = promedio de las últimas N observaciones")
    print("Se usa para suavizar tendencias y reducir ruido")

    # Primero agrupa por fecha
    ventas_por_dia = df.groupby('fecha')['total_venta'].sum().reset_index()
    ventas_por_dia = ventas_por_dia.sort_values('fecha')

    print("\n Ventas por día:")
    print(ventas_por_dia.head(7))

    # TODO 1.5: Calcula el promedio móvil de 3 días
    # Pista: Usa .rolling(window=3).mean()

    # TU CÓDIGO AQUÍ:
    # ventas_por_dia['promedio_movil_3d'] = ventas_por_dia['total_venta'].rolling(window=__).mean()

    # ️ BORRA ESTA LÍNEA cuando hagas tu código:
    ventas_por_dia['promedio_movil_3d'] = 0

    # Verificación (día 3, índice 2)
    if len(ventas_por_dia) >= 3:
        # Calcular manualmente el promedio de los primeros 3 días
        primeros_3 = ventas_por_dia['total_venta'].iloc[0:3]
        esperado_dia_3 = primeros_3.mean()

        if 'promedio_movil_3d' in ventas_por_dia.columns:
            valor_dia_3 = ventas_por_dia.loc[2, 'promedio_movil_3d']
            verificar_feature(
                "Promedio móvil día 3",
                valor_dia_3,
                esperado_dia_3,
                tolerancia=1.0
            )

    print(" Tu resultado:")
    print(ventas_por_dia[['fecha', 'total_venta', 'promedio_movil_3d']].head(7))
    print()

    return ventas_por_dia


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta todos los ejercicios"""
    print("\n" + ""*35)
    print("EJERCICIO 1: FRECUENCIAS Y AGREGACIONES")
    print(""*35 + "\n")

    # Cargar datos
    df = cargar_datos()
    print(" Dataset cargado")
    mostrar_primeras_filas(df, 3)

    # Ejecutar ejercicios
    df = ejercicio_1_1_total_venta(df)
    cantidad_por_producto = ejercicio_1_2_cantidad_total_por_producto(df)
    frecuencia = ejercicio_1_3_frecuencia_productos(df)
    precio_promedio = ejercicio_1_4_precio_promedio_por_categoria(df)
    ventas_por_dia = ejercicio_1_5_promedio_movil(df)

    # Resumen final
    print("\n" + "="*70)
    print(" RESUMEN DE FEATURES CREADAS")
    print("="*70)
    print("\n Completados:")
    print("  1. total_venta (feature derivada)")
    print("  2. cantidad_por_producto (agregación - suma)")
    print("  3. frecuencia_productos (agregación - conteo)")
    print("  4. precio_promedio_categoria (agregación - promedio)")
    print("  5. promedio_movil_3d (ventana deslizante)")

    print("\n PARA APLICAR AL PROYECTO PPT:")
    print("  - total_venta → resultado de cada ronda")
    print("  - cantidad_por_producto → frecuencia de cada jugada")
    print("  - promedio_movil → tendencia reciente del oponente")

    print("\n" + ""*35)
    print("¡Buen trabajo!")
    print(""*35 + "\n")


if __name__ == "__main__":
    main()
