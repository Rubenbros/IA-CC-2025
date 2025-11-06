"""
EJEMPLO 2: Regresión Lineal
============================

Aprende a entrenar modelos de regresión para predecir valores numéricos.

OBJETIVO:
- Entender qué es regresión lineal
- Entrenar un modelo simple
- Evaluar predicciones
- Visualizar resultados
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def ejemplo_regresion_simple():
    """Ejemplo simple: predecir precio según área"""
    print("=" * 70)
    print("EJEMPLO 1: Regresión Lineal Simple")
    print("=" * 70)

    print("\nPROBLEMA: Predecir precio de casas según su área\n")

    # Crear dataset
    print("1. CREAR DATASET")
    print("-" * 70)

    # Datos de casas (área en m², precio en miles €)
    datos = {
        'area_m2': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        'precio_k': [100, 120, 130, 150, 160, 180, 190, 210, 220, 240]
    }

    df = pd.DataFrame(datos)
    print("\nDatos de casas:")
    print(df)

    # Separar X e y
    X = df[['area_m2']]  # Feature (debe ser 2D)
    y = df['precio_k']   # Target

    # División train/test
    print("\n2. DIVIDIR DATOS")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)} casas")
    print(f"Test:  {len(X_test)} casas")

    # Entrenar modelo
    print("\n3. ENTRENAR MODELO")
    print("-" * 70)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    print("Modelo entrenado!")
    print(f"  Coeficiente (pendiente): {modelo.coef_[0]:.2f}")
    print(f"  Intercepto:              {modelo.intercept_:.2f}")
    print(f"\nEcuación: precio = {modelo.coef_[0]:.2f} * area + {modelo.intercept_:.2f}")

    # Hacer predicciones
    print("\n4. HACER PREDICCIONES")
    print("-" * 70)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    print("\nPredicciones en TEST SET:")
    for i, (area, real, pred) in enumerate(zip(X_test['area_m2'], y_test, y_pred_test)):
        print(f"  Casa {i+1}: {area}m² → Predicho: {pred:.1f}k€, Real: {real}k€")

    # Evaluar
    print("\n5. EVALUAR MODELO")
    print("-" * 70)

    # Métricas en train
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Métricas en test
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\nMétricas en TRAIN:")
    print(f"  MSE:      {mse_train:.2f}")
    print(f"  R² Score: {r2_train:.4f} → {r2_train*100:.2f}% de varianza explicada")

    print(f"\nMétricas en TEST:")
    print(f"  MSE:      {mse_test:.2f}")
    print(f"  R² Score: {r2_test:.4f} → {r2_test*100:.2f}% de varianza explicada")

    print("\n INTERPRETACIÓN:")
    if abs(r2_train - r2_test) < 0.1:
        print("  Train y Test similares → Buen ajuste, no hay overfitting")
    else:
        print("  Train y Test muy diferentes → Posible overfitting")

    # Predecir casa nueva
    print("\n6. PREDECIR CASA NUEVA")
    print("-" * 70)

    nueva_casa = [[95]]  # 95 m²
    precio_predicho = modelo.predict(nueva_casa)[0]

    print(f"\nCasa de {nueva_casa[0][0]}m²:")
    print(f"  Precio predicho: {precio_predicho:.1f}k€")

    return modelo, X, y, X_test, y_test, y_pred_test


def ejemplo_regresion_multiple():
    """Regresión con múltiples features"""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Regresión Lineal Múltiple")
    print("=" * 70)

    print("\nPROBLEMA: Predecir salario según experiencia y educación\n")

    # Dataset más complejo
    print("1. CREAR DATASET")
    print("-" * 70)

    datos = {
        'anos_experiencia': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9],
        'anos_educacion': [16, 16, 18, 18, 16, 20, 20, 22, 22, 22, 14, 14, 16, 18, 20],
        'salario_k': [30, 35, 40, 45, 48, 55, 60, 70, 75, 80, 28, 32, 42, 52, 62]
    }

    df = pd.DataFrame(datos)
    print("\nDatos de empleados:")
    print(df.head(10))
    print(f"\nTotal: {len(df)} empleados")

    # Preparar datos
    X = df[['anos_experiencia', 'anos_educacion']]
    y = df['salario_k']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar
    print("\n2. ENTRENAR MODELO")
    print("-" * 70)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    print("Modelo entrenado!")
    print(f"  Coef experiencia: {modelo.coef_[0]:.2f}")
    print(f"  Coef educación:   {modelo.coef_[1]:.2f}")
    print(f"  Intercepto:       {modelo.intercept_:.2f}")

    print(f"\nEcuación:")
    print(f"  salario = {modelo.coef_[0]:.2f} * experiencia + "
          f"{modelo.coef_[1]:.2f} * educacion + {modelo.intercept_:.2f}")

    print("\n INTERPRETACIÓN:")
    print(f"  - Cada año de experiencia añade ~{modelo.coef_[0]:.1f}k€")
    print(f"  - Cada año de educación añade ~{modelo.coef_[1]:.1f}k€")

    # Predecir
    print("\n3. HACER PREDICCIONES")
    print("-" * 70)

    y_pred = modelo.predict(X_test)

    print("\nPredicciones en test set:")
    for exp, edu, real, pred in zip(
        X_test['anos_experiencia'],
        X_test['anos_educacion'],
        y_test,
        y_pred
    ):
        error = abs(real - pred)
        print(f"  Exp: {exp}a, Edu: {edu}a → Predicho: {pred:.1f}k€, "
              f"Real: {real}k€, Error: {error:.1f}k€")

    # Evaluar
    print("\n4. EVALUAR")
    print("-" * 70)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMétricas:")
    print(f"  MAE (Error Absoluto Medio):  {mae:.2f}k€")
    print(f"  RMSE (Raíz del Error Cuad.): {rmse:.2f}k€")
    print(f"  R² Score:                     {r2:.4f} ({r2*100:.2f}%)")

    print(f"\n INTERPRETACIÓN:")
    print(f"  - En promedio nos equivocamos ±{mae:.1f}k€")
    print(f"  - El modelo explica {r2*100:.1f}% de la varianza")

    # Predecir nuevo empleado
    print("\n5. PREDECIR NUEVO EMPLEADO")
    print("-" * 70)

    nuevo_empleado = [[5, 18]]  # 5 años exp, 18 años edu (maestría)
    salario_predicho = modelo.predict(nuevo_empleado)[0]

    print(f"\nEmpleado: 5 años experiencia, 18 años educación")
    print(f"  Salario predicho: {salario_predicho:.1f}k€")


def comparar_metricas():
    """Explica las diferentes métricas de evaluación"""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Entendiendo las Métricas")
    print("=" * 70)

    # Crear predicciones de ejemplo
    y_real = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 290])

    print("\nPredicciones de ejemplo:")
    print(f"  Real:      {y_real}")
    print(f"  Predicho:  {y_pred}")

    # Calcular errores
    errores = y_real - y_pred
    print(f"  Errores:   {errores}")

    # MAE
    print("\n1. MAE (Mean Absolute Error)")
    print("-" * 70)
    mae = mean_absolute_error(y_real, y_pred)
    print(f"  MAE = promedio(|errores|)")
    print(f"  MAE = {mae:.2f}")
    print(f"\n  Interpretación: En promedio nos equivocamos ±{mae:.0f}")
    print(f"  Ventaja: Fácil de entender (mismas unidades que el target)")

    # MSE
    print("\n2. MSE (Mean Squared Error)")
    print("-" * 70)
    mse = mean_squared_error(y_real, y_pred)
    print(f"  MSE = promedio(errores²)")
    print(f"  MSE = {mse:.2f}")
    print(f"\n  Interpretación: Penaliza errores grandes más fuertemente")
    print(f"  Desventaja: Unidades al cuadrado (difícil interpretar)")

    # RMSE
    print("\n3. RMSE (Root Mean Squared Error)")
    print("-" * 70)
    rmse = np.sqrt(mse)
    print(f"  RMSE = √MSE")
    print(f"  RMSE = {rmse:.2f}")
    print(f"\n  Interpretación: Similar a MAE pero penaliza errores grandes")
    print(f"  Ventaja: Mismas unidades que el target")

    # R²
    print("\n4. R² Score (Coeficiente de Determinación)")
    print("-" * 70)
    r2 = r2_score(y_real, y_pred)
    print(f"  R² mide qué % de varianza explica el modelo")
    print(f"  R² = {r2:.4f} = {r2*100:.2f}%")
    print(f"\n  Interpretación:")
    print(f"    1.0 = Predicción perfecta")
    print(f"    0.8-1.0 = Excelente")
    print(f"    0.6-0.8 = Bueno")
    print(f"    0.4-0.6 = Regular")
    print(f"    <0.4 = Malo")
    print(f"    <0.0 = Peor que predecir la media")

    # Comparación
    print("\n CUÁNDO USAR CADA MÉTRICA:")
    print("  - MAE:  Cuando todos los errores importan igual")
    print("  - MSE/RMSE: Cuando errores grandes son muy malos")
    print("  - R²: Para ver qué tan bien ajusta el modelo en general")


def ejemplo_visualizacion(modelo, X, y, X_test, y_test, y_pred_test):
    """Visualiza resultados de regresión"""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Visualización de Resultados")
    print("=" * 70)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gráfico 1: Datos y línea de regresión
        axes[0].scatter(X, y, color='blue', alpha=0.6, label='Datos originales')
        axes[0].plot(X, modelo.predict(X), color='red', linewidth=2, label='Regresión')
        axes[0].scatter(X_test, y_test, color='green', s=100, marker='X',
                       label='Test set', zorder=3)
        axes[0].set_xlabel('Área (m²)')
        axes[0].set_ylabel('Precio (k€)')
        axes[0].set_title('Regresión Lineal: Precio vs Área')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gráfico 2: Real vs Predicho
        axes[1].scatter(y_test, y_pred_test, alpha=0.6)
        axes[1].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Predicción perfecta')
        axes[1].set_xlabel('Precio Real (k€)')
        axes[1].set_ylabel('Precio Predicho (k€)')
        axes[1].set_title('Real vs Predicho')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('regresion_lineal_resultados.png', dpi=100)
        print("\n Gráfico guardado como 'regresion_lineal_resultados.png'")

    except Exception as e:
        print(f"\nNo se pudo generar visualización: {e}")


def errores_comunes():
    """Muestra errores comunes en regresión"""
    print("\n" + "=" * 70)
    print("ERRORES COMUNES EN REGRESIÓN")
    print("=" * 70)

    print("\n ERROR 1: X debe ser 2D")
    print("-" * 70)
    print("  MAL:")
    print("    X = df['area']  # 1D (Serie)")
    print("    modelo.fit(X, y)  # ERROR!")
    print("")
    print("  BIEN:")
    print("    X = df[['area']]  # 2D (DataFrame)")
    print("    modelo.fit(X, y)  # OK")

    print("\n ERROR 2: No normalizar features con escalas diferentes")
    print("-" * 70)
    print("  Problema: area (50-150) vs precio_m2 (1000-3000)")
    print("  Solución: Normalizar con StandardScaler")

    print("\n ERROR 3: No verificar linearidad")
    print("-" * 70)
    print("  Regresión lineal asume relación lineal")
    print("  Si la relación es cuadrática, usar PolynomialFeatures")

    print("\n ERROR 4: Interpretar R² sin contexto")
    print("-" * 70)
    print("  R² = 0.85 puede ser:")
    print("    - Excelente en ciencias sociales")
    print("    - Malo en física/ingeniería")
    print("  Depende del dominio y ruido en los datos")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta todos los ejemplos"""
    print("\n" + "" * 35)
    print("REGRESIÓN LINEAL: ENTRENAR Y EVALUAR")
    print("" * 35)

    # Ejecutar ejemplos
    modelo, X, y, X_test, y_test, y_pred_test = ejemplo_regresion_simple()
    ejemplo_regresion_multiple()
    comparar_metricas()
    ejemplo_visualizacion(modelo, X, y, X_test, y_test, y_pred_test)
    errores_comunes()

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN: REGRESIÓN LINEAL")
    print("=" * 70)

    print("\n1. CUÁNDO USAR:")
    print("  - Predecir valores continuos (precios, temperaturas, etc.)")
    print("  - Relación aproximadamente lineal entre X e y")

    print("\n2. CÓDIGO BÁSICO:")
    print("  from sklearn.linear_model import LinearRegression")
    print("  modelo = LinearRegression()")
    print("  modelo.fit(X_train, y_train)")
    print("  y_pred = modelo.predict(X_test)")

    print("\n3. MÉTRICAS PRINCIPALES:")
    print("  - MAE:  Error promedio en unidades originales")
    print("  - RMSE: Error penalizando grandes errores")
    print("  - R²:   % de varianza explicada (0 a 1)")

    print("\n4. EVALUAR CALIDAD:")
    print("  - Comparar train vs test (detectar overfitting)")
    print("  - R² > 0.7 generalmente bueno")
    print("  - Visualizar real vs predicho")

    print("\n" + "" * 35)


if __name__ == "__main__":
    main()
