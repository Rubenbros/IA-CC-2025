"""
SOLUCIONES - EJERCICIO 1: FRECUENCIAS Y AGREGACIONES
=====================================================

️ ARCHIVO SOLO PARA EL PROFESOR ️

Este archivo contiene las soluciones completas de cada ejercicio.
NO compartir con los alumnos hasta que hayan completado el ejercicio.
"""

import pandas as pd

# =============================================================================
# SOLUCIONES COMPLETAS
# =============================================================================

def solucion_1_1(df):
    """
    EJERCICIO 1.1: Total de venta
    """
    df['total_venta'] = df['cantidad'] * df['precio_unitario']
    return df


def solucion_1_2(df):
    """
    EJERCICIO 1.2: Cantidad total por producto
    """
    cantidad_por_producto = df.groupby('producto')['cantidad'].sum()
    return cantidad_por_producto


def solucion_1_3(df):
    """
    EJERCICIO 1.3: Frecuencia de productos
    """
    conteo = df['producto'].value_counts()
    frecuencia = conteo / len(df)
    return frecuencia


def solucion_1_4(df):
    """
    EJERCICIO 1.4: Precio promedio por categoría
    """
    precio_promedio = df.groupby('categoria')['precio_unitario'].mean()
    return precio_promedio


def solucion_1_5(df):
    """
    EJERCICIO 1.5: Promedio móvil de 3 días
    """
    ventas_por_dia = df.groupby('fecha')['total_venta'].sum().reset_index()
    ventas_por_dia = ventas_por_dia.sort_values('fecha')

    ventas_por_dia['promedio_movil_3d'] = ventas_por_dia['total_venta'].rolling(window=3).mean()

    return ventas_por_dia


# =============================================================================
# CÓDIGO COMPLETO CON TODAS LAS SOLUCIONES
# =============================================================================

def main_completo():
    """Versión completa con todas las soluciones implementadas"""

    # Cargar datos
    df = pd.read_csv('../../src/clase06_feature_engineering/ejercicio_1_ventas.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])

    print("="*70)
    print("SOLUCIONES - EJERCICIO 1")
    print("="*70)

    # 1.1
    print("\n1.1: Total de venta")
    df = solucion_1_1(df)
    print(f" Primera fila: {df.loc[0, 'total_venta']}")

    # 1.2
    print("\n1.2: Cantidad por producto")
    cantidad_por_producto = solucion_1_2(df)
    print(cantidad_por_producto)

    # 1.3
    print("\n1.3: Frecuencia de productos")
    frecuencia = solucion_1_3(df)
    print((frecuencia * 100).round(2))

    # 1.4
    print("\n1.4: Precio promedio por categoría")
    precio_promedio = solucion_1_4(df)
    print(precio_promedio)

    # 1.5
    print("\n1.5: Promedio móvil 3 días")
    ventas_por_dia = solucion_1_5(df)
    print(ventas_por_dia[['fecha', 'total_venta', 'promedio_movil_3d']].head(7))

    print("\n" + "="*70)
    print(" TODAS LAS SOLUCIONES EJECUTADAS")
    print("="*70)


if __name__ == "__main__":
    main_completo()


# =============================================================================
# CÓDIGO PARA COPIAR Y PEGAR (formato compacto)
# =============================================================================
"""
SOLUCIONES RÁPIDAS:

1.1: df['total_venta'] = df['cantidad'] * df['precio_unitario']

1.2: cantidad_por_producto = df.groupby('producto')['cantidad'].sum()

1.3: conteo = df['producto'].value_counts()
     frecuencia = conteo / len(df)

1.4: precio_promedio = df.groupby('categoria')['precio_unitario'].mean()

1.5: ventas_por_dia['promedio_movil_3d'] = ventas_por_dia['total_venta'].rolling(window=3).mean()
"""
