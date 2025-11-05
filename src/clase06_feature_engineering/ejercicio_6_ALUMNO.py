"""
EJERCICIO 6: FEATURES DE SECUENCIAS Y PATRONES
===============================================
⭐⭐⭐ ESTE ES EL MÁS IMPORTANTE PARA PIEDRA, PAPEL O TIJERA ⭐⭐⭐

OBJETIVO: Aprender a extraer features de secuencias (como las jugadas del oponente).

INSTRUCCIONES:
1. Completa donde dice "# TU CÓDIGO AQUÍ"
2. Estas técnicas son EXACTAMENTE las que necesitas para PPT
3. Los conceptos: patrones, entropía, frecuencias

APLICACIÓN DIRECTA A PPT:
- Página visitada → Jugada del oponente
- Secuencia de clics → Secuencia de jugadas
- Entropía → ¿El oponente es predecible?
"""

import pandas as pd
import numpy as np
from collections import Counter

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def cargar_datos():
    df = pd.read_csv('ejercicio_6_clics_usuario.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def verificar_feature(nombre, calculado, esperado, tolerancia=0.1):
    es_correcto = False
    if isinstance(calculado, (int, float)) and isinstance(esperado, (int, float)):
        es_correcto = abs(calculado - esperado) < tolerancia
    else:
        es_correcto = calculado == esperado

    if es_correcto:
        print(f" {nombre}: CORRECTO")
    else:
        print(f" {nombre}: INCORRECTO")
        print(f"   Tu resultado: {calculado}")
        print(f"   Esperado: {esperado}")
    print()


# =============================================================================
# EJERCICIO 6.1: FRECUENCIA DE PÁGINAS (= Frecuencia de jugadas en PPT)
# =============================================================================

def ejercicio_6_1_frecuencia_paginas(df):
    """
    TAREA: Calcular qué porcentaje representa cada página del total

    EN PPT SERÍA: ¿Qué porcentaje de veces el oponente juega piedra, papel, tijera?

    Ejemplo:
    - Usuario visitó: ['inicio', 'productos', 'inicio', 'productos', 'inicio']
    - 'inicio' aparece 3 veces de 5 → frecuencia = 3/5 = 0.6 = 60%
    """
    print("="*70)
    print("EJERCICIO 6.1: Frecuencia de Páginas")
    print("="*70)

    print("\n TEORÍA:")
    print("Frecuencia = cuántas veces aparece / total")
    print("En PPT: Si oponente juega 'piedra' 40% del tiempo, podemos predecir 'papel'")

    # Tomamos un usuario de ejemplo
    usuario = 'user_001'
    df_user = df[df['usuario_id'] == usuario]

    print(f"\n Páginas visitadas por {usuario}:")
    print(df_user['pagina_visitada'].tolist())

    # TODO 6.1: Calcula la frecuencia de cada página para este usuario
    # Pista: Usa value_counts(normalize=True)

    # TU CÓDIGO AQUÍ:
    # frecuencia = df_user['pagina_visitada'].value_counts(normalize=_____)

    # ️ BORRA ESTA LÍNEA:
    frecuencia = pd.Series([0])

    # Verificación
    conteo_manual = df_user['pagina_visitada'].value_counts()
    if 'productos' in conteo_manual.index:
        esperado = conteo_manual['productos'] / len(df_user)
        if 'productos' in frecuencia.index:
            verificar_feature(
                f"Frecuencia 'productos' para {usuario}",
                frecuencia['productos'],
                esperado,
                tolerancia=0.01
            )

    print(" Resultado:")
    if len(frecuencia) > 1:
        print(frecuencia)
    print()

    return frecuencia


# =============================================================================
# EJERCICIO 6.2: BIGRAMAS (Patrones de 2 elementos consecutivos)
# =============================================================================

def ejercicio_6_2_bigramas(df):
    """
    TAREA: Identificar patrones de 2 páginas consecutivas

    EN PPT SERÍA: Si oponente jugó "piedra" → "papel", ¿lo repite?

    Ejemplo:
    - Secuencia: ['inicio', 'productos', 'carrito', 'productos']
    - Bigramas:
        ('inicio', 'productos')
        ('productos', 'carrito')
        ('carrito', 'productos')
    """
    print("="*70)
    print("EJERCICIO 6.2: Bigramas (Patrones de 2)")
    print("="*70)

    print("\n TEORÍA:")
    print("Bigrama = par de elementos consecutivos")
    print("En PPT: Si oponente jugó 'piedra' y luego 'papel', eso es un bigrama")

    usuario = 'user_001'
    df_user = df[df['usuario_id'] == usuario].sort_values('timestamp')
    paginas = df_user['pagina_visitada'].tolist()

    print(f"\n Secuencia de {usuario}:")
    print(paginas)

    # TODO 6.2: Crea una lista de bigramas
    # Pista: Para cada índice i, toma (paginas[i], paginas[i+1])

    bigramas = []
    # TU CÓDIGO AQUÍ:
    # for i in range(len(paginas) - 1):
    #     bigrama = (paginas[i], paginas[_____])
    #     bigramas.append(bigrama)

    # Verificación
    if len(bigramas) > 0:
        primer_bigrama_esperado = (paginas[0], paginas[1])
        verificar_feature(
            "Primer bigrama",
            bigramas[0] if len(bigramas) > 0 else None,
            primer_bigrama_esperado
        )

    print(" Bigramas encontrados:")
    if len(bigramas) > 0:
        for bigrama in bigramas[:5]:
            print(f"   {bigrama[0]} → {bigrama[1]}")
    print()

    # Contar frecuencia de bigramas
    if len(bigramas) > 0:
        contador = Counter(bigramas)
        print(" Bigrama más común:")
        mas_comun = contador.most_common(1)[0]
        print(f"   {mas_comun[0][0]} → {mas_comun[0][1]} (aparece {mas_comun[1]} veces)")
        print()

    return bigramas


# =============================================================================
# EJERCICIO 6.3: ENTROPÍA (¿Qué tan predecible es?)
# =============================================================================

def ejercicio_6_3_entropia(df):
    """
    TAREA: Calcular la entropía (aleatoriedad) del comportamiento

    EN PPT: ¿El oponente es predecible o aleatorio?

    Formula de Shannon: H = -Σ p(x) * log2(p(x))

    Interpretación:
    - H ≈ 0: Muy predecible (siempre hace lo mismo)
    - H ≈ 1.58: Muy aleatorio (distribución uniforme entre 3 opciones)
    """
    print("="*70)
    print("EJERCICIO 6.3: Entropía (Predictibilidad)")
    print("="*70)

    print("\n TEORÍA:")
    print("Entropía mide aleatoriedad:")
    print("  - Baja entropía = predecible")
    print("  - Alta entropía = aleatorio")
    print("\nEn PPT: Si oponente siempre juega piedra → entropía ≈ 0")

    usuario = 'user_001'
    df_user = df[df['usuario_id'] == usuario]
    paginas = df_user['pagina_visitada'].tolist()

    print(f"\n Páginas de {usuario}:")
    print(paginas)

    # TODO 6.3: Calcula la entropía
    # PASO 1: Calcula las frecuencias (probabilidades)
    # TU CÓDIGO AQUÍ:
    # frecuencias = df_user['pagina_visitada'].value_counts(normalize=True)
    # probabilidades = frecuencias.values

    # ️ BORRA ESTA LÍNEA:
    probabilidades = np.array([0.5, 0.5])

    # PASO 2: Aplica la formula de entropía
    # H = -Σ p * log2(p)
    # TU CÓDIGO AQUÍ:
    # entropia = -np.sum(probabilidades * np.log2(probabilidades))

    # ️ BORRA ESTA LÍNEA:
    entropia = 0

    # Verificación (calculamos la correcta)
    frec_correcta = df_user['pagina_visitada'].value_counts(normalize=True).values
    entropia_esperada = -np.sum(frec_correcta * np.log2(frec_correcta + 1e-10))

    verificar_feature(f"Entropía de {usuario}", entropia, entropia_esperada, tolerancia=0.1)

    print(" Interpretación:")
    if entropia < 1.0:
        print("   → Usuario PREDECIBLE (sigue patrones claros)")
    elif entropia < 1.5:
        print("   → Usuario ALGO PREDECIBLE")
    else:
        print("   → Usuario ALEATORIO (difícil de predecir)")
    print()

    return entropia


# =============================================================================
# EJERCICIO 6.4: LAG FEATURE (Página anterior)
# =============================================================================

def ejercicio_6_4_lag_feature(df):
    """
    TAREA: Crear una columna con la página visitada anteriormente

    EN PPT: Saber qué jugó el oponente en la ronda anterior

    Función: .shift(1) desplaza los valores una posición
    """
    print("="*70)
    print("EJERCICIO 6.4: Lag Feature (Valor Anterior)")
    print("="*70)

    print("\n TEORÍA:")
    print("Lag = valor pasado")
    print("shift(1) = desplazar una posición hacia abajo")
    print("\nEn PPT: La jugada anterior del oponente puede influir en la siguiente")

    usuario = 'user_001'
    df_user = df[df['usuario_id'] == usuario].sort_values('timestamp').copy()

    print(f"\n Secuencia original de {usuario}:")
    print(df_user['pagina_visitada'].tolist())

    # TODO 6.4: Crea una columna con la página anterior
    # Pista: Usa .shift(1)

    # TU CÓDIGO AQUÍ:
    # df_user['pagina_anterior'] = df_user['pagina_visitada'].shift(___)

    # ️ BORRA ESTA LÍNEA:
    df_user['pagina_anterior'] = None

    # Verificación
    if 'pagina_anterior' in df_user.columns:
        # La segunda fila debe tener como "anterior" la primera página
        primera_pagina = df_user['pagina_visitada'].iloc[0]
        anterior_segunda_fila = df_user['pagina_anterior'].iloc[1]

        verificar_feature(
            "Página anterior (segunda fila)",
            anterior_segunda_fila,
            primera_pagina
        )

        print(" Resultado:")
        print(df_user[['pagina_visitada', 'pagina_anterior']].head())
        print()

    return df_user


# =============================================================================
# EJERCICIO 6.5: VENTANA DESLIZANTE (Últimas 3 acciones)
# =============================================================================

def ejercicio_6_5_ventana(df):
    """
    TAREA: Contar cuántos 'clics' hubo en las últimas 3 acciones

    EN PPT: Contar cuántas veces jugó 'piedra' en las últimas 3 rondas

    Función: .rolling(window=3)
    """
    print("="*70)
    print("EJERCICIO 6.5: Ventana Deslizante (Últimas 3)")
    print("="*70)

    print("\n TEORÍA:")
    print("Ventana = considerar solo los últimos N elementos")
    print("Útil para capturar tendencias recientes")
    print("\nEn PPT: '¿Cuántas veces jugó piedra en las últimas 3 rondas?'")

    usuario = 'user_001'
    df_user = df[df['usuario_id'] == usuario].sort_values('timestamp').copy()

    # Crear columna binaria: 1 si acción es 'clic', 0 si no
    df_user['es_clic'] = (df_user['accion'] == 'clic').astype(int)

    print(f"\n Acciones de {usuario}:")
    print(df_user[['accion', 'es_clic']].head(6))

    # TODO 6.5: Cuenta cuántos clics hay en ventana de 3
    # Pista: Usa .rolling(window=3).sum()

    # TU CÓDIGO AQUÍ:
    # df_user['clics_ultimas_3'] = df_user['es_clic'].rolling(window=___).sum()

    # ️ BORRA ESTA LÍNEA:
    df_user['clics_ultimas_3'] = 0

    # Verificación (fila 4, índice 3 - debe sumar las 3 anteriores)
    if len(df_user) >= 4 and 'clics_ultimas_3' in df_user.columns:
        suma_manual = df_user['es_clic'].iloc[1:4].sum()  # Filas 1, 2, 3
        valor_calculado = df_user['clics_ultimas_3'].iloc[3]

        verificar_feature(
            "Clics en ventana 3 (fila 4)",
            valor_calculado,
            suma_manual,
            tolerancia=0.1
        )

    print(" Resultado:")
    print(df_user[['accion', 'es_clic', 'clics_ultimas_3']].head(7))
    print()

    return df_user


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    print("\n" + ""*35)
    print("EJERCICIO 6: SECUENCIAS Y PATRONES")
    print("⭐ APLICACIÓN DIRECTA A PIEDRA, PAPEL O TIJERA ⭐")
    print(""*35 + "\n")

    df = cargar_datos()
    print(" Dataset cargado: Navegación de usuarios\n")

    frecuencia = ejercicio_6_1_frecuencia_paginas(df)
    bigramas = ejercicio_6_2_bigramas(df)
    entropia = ejercicio_6_3_entropia(df)
    df_lag = ejercicio_6_4_lag_feature(df)
    df_ventana = ejercicio_6_5_ventana(df)

    print("="*70)
    print(" APLICACIÓN AL PROYECTO PPT")
    print("="*70)
    print("\nLo que aprendiste aquí es EXACTAMENTE lo que necesitas:")
    print("\n1. FRECUENCIAS → ¿Qué jugada prefiere el oponente?")
    print("   Código PPT: frecuencia_piedra = (jugadas == 'piedra').sum() / len(jugadas)")
    print("\n2. BIGRAMAS → ¿Qué patrón de 2 jugadas repite?")
    print("   Código PPT: for i in range(len(jugadas)-1):")
    print("                   bigrama = (jugadas[i], jugadas[i+1])")
    print("\n3. ENTROPÍA → ¿Es predecible o aleatorio?")
    print("   Código PPT: H = -sum(p * log2(p)) donde p son las frecuencias")
    print("\n4. LAG → ¿Qué jugó en la ronda anterior?")
    print("   Código PPT: jugada_anterior = jugadas.shift(1)")
    print("\n5. VENTANA → ¿Cuántas veces jugó X en las últimas N rondas?")
    print("   Código PPT: ultimas_3 = jugadas[-3:]")

    print("\n" + "="*70)
    print(" CONSEJO PARA PPT:")
    print("="*70)
    print("\nCombina todas estas features en tu agente:")
    print("  - Si entropía baja → Usa frecuencias")
    print("  - Si entropía alta → Juega aleatorio")
    print("  - Si detectas bigramas → Predice basado en patrón")
    print("  - Siempre incluye lag features (jugadas anteriores)")

    print("\n" + ""*35 + "\n")


if __name__ == "__main__":
    main()
