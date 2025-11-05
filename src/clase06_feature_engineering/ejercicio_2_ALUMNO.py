"""
EJERCICIO 2: TRANSFORMACIONES MATEMÁTICAS
==========================================

OBJETIVO: Aprender a aplicar transformaciones para mejorar features numéricas.

INSTRUCCIONES:
1. Completa el código donde dice "# TU CÓDIGO AQUÍ"
2. Las funciones te dirán si está bien o mal
3. Solo necesitas operaciones matemáticas básicas y funciones de numpy

TRANSFORMACIONES QUE APRENDERÁS:
- Logaritmo (para datos con crecimiento exponencial)
- Raíz cuadrada (para reducir outliers)
- Normalización (escalar a rango 0-1)
- Features derivadas (combinaciones)
"""

import pandas as pd
import numpy as np

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def cargar_datos():
    df = pd.read_csv('ejercicio_2_casas.csv')
    return df


def verificar_feature(nombre, calculado, esperado, tolerancia=0.1):
    if abs(calculado - esperado) < tolerancia:
        print(f" {nombre}: CORRECTO ({calculado:.2f})")
    else:
        print(f" {nombre}: INCORRECTO")
        print(f"   Tu resultado: {calculado:.2f}")
        print(f"   Esperado: {esperado:.2f}")
    print()


# =============================================================================
# EJERCICIO 2.1: FEATURE DERIVADA - PRECIO POR M²
# =============================================================================

def ejercicio_2_1_precio_por_m2(df):
    """
    TAREA: Calcular el precio por metro cuadrado

    Formula: precio_por_m2 = precio_euros / area_m2

    Ejemplo:
    - Precio: 120,000€
    - Área: 45 m²
    - Precio/m²: 120,000 / 45 = 2,666.67 €/m²
    """
    print("="*70)
    print("EJERCICIO 2.1: Precio por Metro Cuadrado")
    print("="*70)

    print("\n TEORÍA:")
    print("Esta feature nos dice cuán caro es el m² de cada casa.")
    print("Útil para comparar casas de diferentes tamaños.")

    print("\n Ejemplo:")
    print(df[['area_m2', 'precio_euros']].head(3))

    # TODO 2.1: Calcula precio_por_m2
    # Pista: Divide precio_euros entre area_m2

    # TU CÓDIGO AQUÍ:
    # df['precio_por_m2'] = df['_______'] / df['area_m2']

    # ️ BORRA ESTA LÍNEA:
    df['precio_por_m2'] = 0

    # Verificación
    esperado = 120000 / 45  # Primera fila
    if 'precio_por_m2' in df.columns:
        verificar_feature("Precio/m² (primera casa)", df.loc[0, 'precio_por_m2'], esperado, tolerancia=10)

    return df


# =============================================================================
# EJERCICIO 2.2: TRANSFORMACIÓN LOGARÍTMICA
# =============================================================================

def ejercicio_2_2_logaritmo(df):
    """
    TAREA: Aplicar logaritmo al precio

    ¿Por qué? Los precios tienen distribución sesgada (algunos muy altos).
    El logaritmo "comprime" los valores grandes.

    Función: np.log(valor)

    Ejemplo:
    - Precio: 120,000 → log(120,000) ≈ 11.70
    - Precio: 450,000 → log(450,000) ≈ 13.02
    """
    print("="*70)
    print("EJERCICIO 2.2: Transformación Logarítmica")
    print("="*70)

    print("\n TEORÍA:")
    print("Logaritmo: convierte multiplicaciones en sumas")
    print("Ejemplo: 10 vs 100 vs 1000 → log: 1 vs 2 vs 3 (más uniforme)")

    print("\n Precios originales:")
    print(df['precio_euros'].head())

    # TODO 2.2: Aplica logaritmo al precio
    # Pista: Usa np.log()

    # TU CÓDIGO AQUÍ:
    # df['log_precio'] = np.log(df['_______'])

    # ️ BORRA ESTA LÍNEA:
    df['log_precio'] = 0

    # Verificación
    esperado = np.log(120000)
    if 'log_precio' in df.columns:
        verificar_feature("Log precio (primera casa)", df.loc[0, 'log_precio'], esperado, tolerancia=0.01)

    if df['log_precio'].sum() > 0:
        print(" Comparación:")
        print(df[['precio_euros', 'log_precio']].head())
        print()

    return df


# =============================================================================
# EJERCICIO 2.3: RAÍZ CUADRADA
# =============================================================================

def ejercicio_2_3_raiz_cuadrada(df):
    """
    TAREA: Aplicar raíz cuadrada a la distancia al centro

    ¿Por qué? Reduce el impacto de valores muy grandes.
    Menos agresivo que el logaritmo.

    Función: np.sqrt(valor)

    Ejemplo:
    - Distancia: 16 km → √16 = 4
    - Distancia: 4 km → √4 = 2
    """
    print("="*70)
    print("EJERCICIO 2.3: Raíz Cuadrada")
    print("="*70)

    print("\n TEORÍA:")
    print("Raíz cuadrada: reduce diferencias entre valores grandes")
    print("Útil cuando hay outliers moderados")

    # TODO 2.3: Aplica raíz cuadrada a distancia_centro_km
    # Pista: Usa np.sqrt()

    # TU CÓDIGO AQUÍ:
    # df['sqrt_distancia'] = np.sqrt(df['_______'])

    # ️ BORRA ESTA LÍNEA:
    df['sqrt_distancia'] = 0

    # Verificación
    esperado = np.sqrt(12.5)
    if 'sqrt_distancia' in df.columns:
        verificar_feature("√distancia (primera casa)", df.loc[0, 'sqrt_distancia'], esperado, tolerancia=0.01)

    return df


# =============================================================================
# EJERCICIO 2.4: NORMALIZACIÓN MIN-MAX
# =============================================================================

def ejercicio_2_4_normalizacion(df):
    """
    TAREA: Normalizar el área a rango [0, 1]

    Formula: valor_normalizado = (valor - min) / (max - min)

    Ejemplo:
    - Área mínima: 45 m²
    - Área máxima: 200 m²
    - Área actual: 120 m²
    - Normalizado: (120 - 45) / (200 - 45) = 75 / 155 = 0.48
    """
    print("="*70)
    print("EJERCICIO 2.4: Normalización Min-Max [0, 1]")
    print("="*70)

    print("\n TEORÍA:")
    print("Normalización: escala valores al rango [0, 1]")
    print("Útil para algoritmos sensibles a la escala (KNN, Redes Neuronales)")

    area_min = df['area_m2'].min()
    area_max = df['area_m2'].max()

    print(f"\n Info:")
    print(f"Área mínima: {area_min} m²")
    print(f"Área máxima: {area_max} m²")

    # TODO 2.4: Normaliza el área
    # Formula: (valor - min) / (max - min)

    # TU CÓDIGO AQUÍ:
    # df['area_normalizada'] = (df['area_m2'] - _____) / (_____ - _____)

    # ️ BORRA ESTA LÍNEA:
    df['area_normalizada'] = 0

    # Verificación
    esperado = (45 - area_min) / (area_max - area_min)
    if 'area_normalizada' in df.columns:
        verificar_feature("Área normalizada (primera casa)", df.loc[0, 'area_normalizada'], esperado, tolerancia=0.01)

        print(f" Rango después de normalización:")
        print(f"   Mínimo: {df['area_normalizada'].min():.2f}")
        print(f"   Máximo: {df['area_normalizada'].max():.2f}")
        print()

    return df


# =============================================================================
# EJERCICIO 2.5: FEATURE DE INTERACCIÓN
# =============================================================================

def ejercicio_2_5_interaccion(df):
    """
    TAREA: Crear un "score de ubicación" inverso a la distancia

    Formula: score = 1 / (1 + distancia_centro_km)

    ¿Por qué? Casas más cercanas al centro tienen score más alto.

    Ejemplo:
    - Distancia: 5 km → Score: 1/(1+5) = 0.167
    - Distancia: 15 km → Score: 1/(1+15) = 0.063
    """
    print("="*70)
    print("EJERCICIO 2.5: Score de Ubicación")
    print("="*70)

    print("\n TEORÍA:")
    print("Feature de interacción: combina información de otra feature")
    print("Inverso: valores pequeños (cerca) → score alto")

    # TODO 2.5: Calcula el score de ubicación
    # Pista: 1 / (1 + distancia)

    # TU CÓDIGO AQUÍ:
    # df['score_ubicacion'] = 1 / (1 + df['_______'])

    # ️ BORRA ESTA LÍNEA:
    df['score_ubicacion'] = 0

    # Verificación
    esperado = 1 / (1 + 12.5)
    if 'score_ubicacion' in df.columns:
        verificar_feature("Score ubicación (primera casa)", df.loc[0, 'score_ubicacion'], esperado, tolerancia=0.01)

        print(" Interpretación:")
        print("   Score alto (>0.2): Cerca del centro")
        print("   Score bajo (<0.1): Lejos del centro")
        print()

    return df


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    print("\n" + ""*35)
    print("EJERCICIO 2: TRANSFORMACIONES MATEMÁTICAS")
    print(""*35 + "\n")

    df = cargar_datos()
    print(" Dataset cargado: 30 casas\n")

    df = ejercicio_2_1_precio_por_m2(df)
    df = ejercicio_2_2_logaritmo(df)
    df = ejercicio_2_3_raiz_cuadrada(df)
    df = ejercicio_2_4_normalizacion(df)
    df = ejercicio_2_5_interaccion(df)

    print("="*70)
    print(" RESUMEN")
    print("="*70)
    print("\n Features creadas:")
    print("  1. precio_por_m2 (derivada)")
    print("  2. log_precio (transformación)")
    print("  3. sqrt_distancia (transformación)")
    print("  4. area_normalizada (escalado)")
    print("  5. score_ubicacion (interacción)")

    print("\n CUÁNDO USAR CADA UNA:")
    print("  - Log: Datos muy sesgados (precios, ingresos)")
    print("  - √: Outliers moderados")
    print("  - Normalización: Algoritmos sensibles a escala")
    print("  - Score/interacción: Combinar información")

    # Guardar
    df.to_csv('ejercicio_2_resultado.csv', index=False)
    print("\n Resultado guardado en 'ejercicio_2_resultado.csv'")

    print("\n" + ""*35 + "\n")


if __name__ == "__main__":
    main()
