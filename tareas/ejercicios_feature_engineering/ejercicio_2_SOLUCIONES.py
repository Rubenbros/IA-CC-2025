"""
SOLUCIONES - EJERCICIO 2: TRANSFORMACIONES MATEMÁTICAS
=======================================================

️ ARCHIVO SOLO PARA EL PROFESOR ️

Este archivo contiene las soluciones completas de cada ejercicio.
NO compartir con los alumnos hasta que hayan completado el ejercicio.
"""

import pandas as pd
import numpy as np

# =============================================================================
# SOLUCIONES COMPLETAS
# =============================================================================

def solucion_2_1(df):
    """
    EJERCICIO 2.1: Precio por metro cuadrado
    """
    df['precio_por_m2'] = df['precio_euros'] / df['area_m2']
    return df


def solucion_2_2(df):
    """
    EJERCICIO 2.2: Transformación logarítmica
    """
    df['log_precio'] = np.log(df['precio_euros'])
    return df


def solucion_2_3(df):
    """
    EJERCICIO 2.3: Raíz cuadrada
    """
    df['sqrt_distancia'] = np.sqrt(df['distancia_centro_km'])
    return df


def solucion_2_4(df):
    """
    EJERCICIO 2.4: Normalización Min-Max
    """
    area_min = df['area_m2'].min()
    area_max = df['area_m2'].max()

    df['area_normalizada'] = (df['area_m2'] - area_min) / (area_max - area_min)
    return df


def solucion_2_5(df):
    """
    EJERCICIO 2.5: Score de ubicación
    """
    df['score_ubicacion'] = 1 / (1 + df['distancia_centro_km'])
    return df


# =============================================================================
# CÓDIGO COMPLETO CON TODAS LAS SOLUCIONES
# =============================================================================

def main_completo():
    """Versión completa con todas las soluciones implementadas"""

    # Cargar datos
    df = pd.read_csv('../../src/clase06_feature_engineering/ejercicio_2_casas.csv')

    print("="*70)
    print("SOLUCIONES - EJERCICIO 2")
    print("="*70)

    # 2.1
    print("\n2.1: Precio por m²")
    df = solucion_2_1(df)
    print(f" Primera casa: {df.loc[0, 'precio_por_m2']:.2f} €/m²")

    # 2.2
    print("\n2.2: Logaritmo del precio")
    df = solucion_2_2(df)
    print(f" Primera casa: log({df.loc[0, 'precio_euros']}) = {df.loc[0, 'log_precio']:.2f}")

    # 2.3
    print("\n2.3: Raíz cuadrada de distancia")
    df = solucion_2_3(df)
    print(f" Primera casa: √{df.loc[0, 'distancia_centro_km']:.1f} = {df.loc[0, 'sqrt_distancia']:.2f}")

    # 2.4
    print("\n2.4: Normalización del área")
    df = solucion_2_4(df)
    print(f" Rango: [{df['area_normalizada'].min():.2f}, {df['area_normalizada'].max():.2f}]")

    # 2.5
    print("\n2.5: Score de ubicación")
    df = solucion_2_5(df)
    print(f" Primera casa: {df.loc[0, 'score_ubicacion']:.3f}")

    print("\n" + "="*70)
    print(" TODAS LAS SOLUCIONES EJECUTADAS")
    print("="*70)

    # Guardar resultado
    df.to_csv('ejercicio_2_resultado_completo.csv', index=False)
    print(" Guardado en: ejercicio_2_resultado_completo.csv")


if __name__ == "__main__":
    main_completo()


# =============================================================================
# CÓDIGO PARA COPIAR Y PEGAR (formato compacto)
# =============================================================================
"""
SOLUCIONES RÁPIDAS:

2.1: df['precio_por_m2'] = df['precio_euros'] / df['area_m2']

2.2: df['log_precio'] = np.log(df['precio_euros'])

2.3: df['sqrt_distancia'] = np.sqrt(df['distancia_centro_km'])

2.4: area_min = df['area_m2'].min()
     area_max = df['area_m2'].max()
     df['area_normalizada'] = (df['area_m2'] - area_min) / (area_max - area_min)

2.5: df['score_ubicacion'] = 1 / (1 + df['distancia_centro_km'])
"""
