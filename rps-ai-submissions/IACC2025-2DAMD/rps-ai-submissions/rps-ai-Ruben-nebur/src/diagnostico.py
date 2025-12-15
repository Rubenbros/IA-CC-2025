"""
Script de diagnóstico para detectar desbalanceo en el modelo RPSAI
Ejecuta: python diagnostico.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"
RUTA_SCALER = RUTA_PROYECTO / "models" / "scaler.pkl"

# Importar funciones del modelo
from modelo import cargar_datos, preparar_datos, crear_features, seleccionar_features


def cargar_modelo():
    """Carga el modelo entrenado"""
    with open(RUTA_MODELO, "rb") as f:
        modelo = pickle.load(f)
    with open(RUTA_SCALER, "rb") as f:
        scaler = pickle.load(f)
    return modelo, scaler


def diagnostico_completo():
    """Ejecuta diagnóstico completo del modelo"""

    print("\n" + "=" * 60)
    print("   🔍 DIAGNÓSTICO DE DESBALANCEO - RPSAI")
    print("=" * 60)

    # Cargar datos
    print("\n📂 Cargando datos...")
    df = cargar_datos()
    df = preparar_datos(df)
    df = crear_features(df)
    X, y = seleccionar_features(df)

    # Cargar modelo
    print("🤖 Cargando modelo entrenado...")
    modelo, scaler = cargar_modelo()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # ==========================================
    # 1. DATASET ORIGINAL
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 1. DISTRIBUCIÓN EN EL DATASET ORIGINAL")
    print("=" * 60)

    jugadas = df['jugada_j2'].value_counts()
    total = len(df)

    distribuciones = {}
    for jugada in ['piedra', 'papel', 'tijera']:
        count = jugadas.get(jugada, 0)
        pct = (count / total) * 100
        distribuciones[jugada] = pct

        # Barra visual
        barra = "█" * int(pct / 2)
        estado = "✓" if 25 <= pct <= 42 else "⚠"

        print(f"{estado} {jugada.capitalize():8} │ {barra:<20} │ {count:4} ({pct:5.1f}%)")

    # ==========================================
    # 2. PREDICCIONES DEL MODELO
    # ==========================================
    print("\n" + "=" * 60)
    print("🤖 2. PREDICCIONES DEL MODELO EN TEST")
    print("=" * 60)

    X_test_scaled = scaler.transform(X_test)
    y_pred = modelo.predict(X_test_scaled)

    pred_counts = pd.Series(y_pred).value_counts()
    total_pred = len(y_pred)

    predicciones = {}
    nombres = ['piedra', 'papel', 'tijera']

    for i, jugada in enumerate(nombres):
        count = pred_counts.get(float(i), 0)
        pct = (count / total_pred) * 100
        predicciones[jugada] = pct

        # Barra visual
        barra = "█" * int(pct / 2)

        # Estado: verde si está cerca de 33%, rojo si muy desviado
        desviacion = abs(pct - 33.33)
        if desviacion > 15:
            estado = "🔴"
            alerta = " ⚠ SESGO ALTO"
        elif desviacion > 8:
            estado = "🟡"
            alerta = " ⚠ Sesgo moderado"
        else:
            estado = "🟢"
            alerta = ""

        print(f"{estado} {jugada.capitalize():8} │ {barra:<20} │ {count:4} ({pct:5.1f}%){alerta}")

    # ==========================================
    # 3. COMPARACIÓN
    # ==========================================
    print("\n" + "=" * 60)
    print("📈 3. COMPARACIÓN: DATASET vs PREDICCIONES")
    print("=" * 60)

    print(f"{'':12} │ {'Dataset':>10} │ {'Modelo':>10} │ {'Diferencia':>12}")
    print("-" * 60)

    for jugada in ['piedra', 'papel', 'tijera']:
        dataset_pct = distribuciones[jugada]
        modelo_pct = predicciones[jugada]
        diff = modelo_pct - dataset_pct

        # Indicador de dirección
        if abs(diff) < 3:
            indicador = "  ≈"
        elif diff > 0:
            indicador = "  ↑"
        else:
            indicador = "  ↓"

        print(f"{jugada.capitalize():12} │ {dataset_pct:9.1f}% │ {modelo_pct:9.1f}% │ {diff:+10.1f}%{indicador}")

    # ==========================================
    # 4. MATRIZ DE CONFUSIÓN
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 4. MATRIZ DE CONFUSIÓN")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_pred)

    print("\n              PREDICCIÓN")
    print("        │  Piedra   Papel  Tijera │ Total")
    print("-" * 60)

    for i, label in enumerate(['Piedra', 'Papel', 'Tijera']):
        print(f"{label:7} │", end="")

        for j in range(3):
            valor = cm[i][j]
            if i == j:
                # Diagonal (aciertos)
                print(f"   {valor:3}✓  ", end="")
            else:
                print(f"   {valor:3}   ", end="")

        total_fila = cm[i].sum()
        accuracy_fila = (cm[i][i] / total_fila * 100) if total_fila > 0 else 0
        print(f"│ {total_fila:3} ({accuracy_fila:4.1f}%)")

    # ==========================================
    # 5. MÉTRICAS POR CLASE
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 5. MÉTRICAS DETALLADAS POR CLASE")
    print("=" * 60)

    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    print(f"\n{'Clase':10} │ {'Precision':>10} │ {'Recall':>10} │ {'F1-Score':>10}")
    print("-" * 60)

    for i, jugada in enumerate(['Piedra', 'Papel', 'Tijera']):
        print(f"{jugada:10} │ {precision[i]:9.3f}  │ {recall[i]:9.3f}  │ {f1[i]:9.3f}")

    # ==========================================
    # 6. DIAGNÓSTICO Y RECOMENDACIONES
    # ==========================================
    print("\n" + "=" * 60)
    print("💡 6. DIAGNÓSTICO Y RECOMENDACIONES")
    print("=" * 60)

    # Detectar problemas
    problemas = []

    for jugada in ['piedra', 'papel', 'tijera']:
        desv = abs(predicciones[jugada] - 33.33)
        if desv > 15:
            problemas.append(f"- Sesgo ALTO hacia '{jugada}' ({predicciones[jugada]:.1f}%)")
        elif desv > 8:
            problemas.append(f"- Sesgo moderado hacia '{jugada}' ({predicciones[jugada]:.1f}%)")

    if problemas:
        print("\n⚠️  PROBLEMAS DETECTADOS:\n")
        for p in problemas:
            print(f"  {p}")

        print("\n🔧 SOLUCIONES RECOMENDADAS:\n")
        print("  1. Añadir class_weight='balanced' en DecisionTree y RandomForest")
        print("  2. Si el dataset está desbalanceado, recolectar más datos equilibrados")
        print("  3. Implementar aleatoriedad estratégica (80% modelo, 20% random)")
        print("  4. Usar ensemble con voting para suavizar predicciones")

    else:
        print("\n✅ El modelo está BALANCEADO")
        print("   Las predicciones están distribuidas equilibradamente.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        diagnostico_completo()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de haber entrenado el modelo primero:")
        print("  python src/modelo.py")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback

        traceback.print_exc()