# Feature Engineering: De Datos Crudos a Features Predictivas

## Índice
1. [Introducción al Feature Engineering](#1-introducción)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
3. [Tipos de Features](#3-tipos-de-features)
4. [Técnicas de Feature Engineering](#4-técnicas-de-feature-engineering)
5. [Feature Engineering para Piedra, Papel o Tijera](#5-aplicación-al-proyecto-ppt)
6. [Validación y Selección de Features](#6-validación-y-selección)
7. [Errores Comunes y Mejores Prácticas](#7-mejores-prácticas)
8. [Ejercicios Propuestos](#8-ejercicios)

---

## 1. Introducción al Feature Engineering

### ¿Qué es Feature Engineering?

**Feature Engineering** (Ingeniería de Características) es el proceso de usar conocimiento del dominio para crear nuevas variables (features) a partir de datos crudos que ayuden a los algoritmos de Machine Learning a aprender mejor.

> "Los datos vienen en bruto, pero los modelos necesitan información procesada"

### ¿Por qué es importante?

```
Datos Crudos → Feature Engineering → Modelo ML → Predicciones

Sin FE:    [jugada anterior] → Modelo → 40% accuracy
Con FE:    [jugada anterior, frecuencia, patrón, racha] → Modelo → 60% accuracy
```

**Impacto real:**
- Puede mejorar el rendimiento del modelo en 10-50%
- A menudo más importante que elegir el algoritmo "perfecto"
- Distingue a un científico de datos junior de uno senior

### La jerarquía de importancia en ML

```
1. Datos de calidad (40%)
2. Feature Engineering (30%)
3. Elección del modelo (20%)
4. Hiperparámetros (10%)
```

### Ejemplo motivador: Predecir la temperatura

**Datos crudos:**
```
fecha: 2024-10-30
temperatura: 18°C
```

**Con feature engineering:**
```
fecha: 2024-10-30
temperatura: 18°C
mes: octubre              ← Extraído de fecha
estación: otoño          ← Derivado de mes
día_semana: miércoles    ← Extraído de fecha
es_fin_semana: False     ← Derivado de día_semana
temp_media_7dias: 16°C   ← Agregación temporal
```

El segundo conjunto permite al modelo aprender patrones estacionales y semanales.

---

## 2. Fundamentos Teóricos

### 2.1 El Problema de Representación

Los algoritmos de ML trabajan con números, pero la realidad es compleja. Necesitamos **representar** la realidad de forma que los algoritmos puedan entenderla.

**Teorema No Free Lunch:**
> No existe un algoritmo universalmente mejor para todos los problemas. El rendimiento depende de cómo representamos los datos.

### 2.2 El Espacio de Features

Cada feature es una **dimensión** en el espacio de representación:

```
1D: [frecuencia_piedra]
    Modelo puede detectar: preferencia simple

2D: [frecuencia_piedra, frecuencia_papel]
    Modelo puede detectar: relación entre dos preferencias

3D: [frecuencia_piedra, frecuencia_papel, frecuencia_tijera]
    Modelo puede detectar: distribución completa de preferencias

10D: [frecuencias + patrones + rachas + tiempo]
    Modelo puede detectar: comportamiento complejo
```

### 2.3 Información Mutua

La **información mutua** I(X;Y) mide cuánta información proporciona la feature X sobre la variable objetivo Y:

```
I(X;Y) = 0  →  X no ayuda a predecir Y (inútil)
I(X;Y) alto →  X es muy predictiva de Y (valiosa)
```

**Ejemplo en PPT:**
```
jugada_anterior → jugada_siguiente: I = 0.15 (baja, jugadores varían)
patrón_3_jugadas → jugada_siguiente: I = 0.35 (media, mejor predictor)
```

### 2.4 La Maldición de la Dimensionalidad

Añadir más features no siempre es mejor:

```
N = 100 muestras

Con 3 features:  Densidad = 100/3³ = 3.7 muestras por región
Con 10 features: Densidad = 100/10¹⁰ = 0.0000001 (casi vacío!)
```

**Consecuencias:**
- Más features requieren exponencialmente más datos
- El modelo puede memorizar en lugar de generalizar (overfitting)
- Algunos features pueden ser ruido que confunda al modelo

**Principio de Parsimonia (Navaja de Occam):**
> Entre dos modelos con similar rendimiento, elige el más simple

---

## 3. Tipos de Features

### 3.1 Features Básicas (Directas)

Datos tal como vienen en el dataset:

```python
# Ejemplo PPT
jugada_jugador = "piedra"
jugada_oponente = "papel"
numero_ronda = 5
resultado = "derrota"
```

**Limitaciones:**
- No capturan contexto
- No expresan relaciones
- Limitada capacidad predictiva

### 3.2 Features Derivadas (Transformaciones)

Creadas mediante operaciones matemáticas o lógicas:

**Matemáticas:**
```python
# Original
velocidad = 60  # km/h
tiempo = 2      # horas

# Derivada
distancia = velocidad * tiempo  # 120 km
```

**Lógicas:**
```python
# Original
edad = 17

# Derivada
es_mayor_edad = edad >= 18  # False
categoria_edad = "adolescente"  # if 13 <= edad < 18
```

**En PPT:**
```python
# Original
jugadas = ["piedra", "papel", "piedra"]

# Derivadas
total_jugadas = len(jugadas)  # 3
ultima_jugada = jugadas[-1]   # "piedra"
jugadas_repetidas = (jugadas[-1] == jugadas[-2])  # True
```

### 3.3 Features de Agregación

Resumen estadístico de múltiples valores:

**Estadísticos básicos:**
```python
# Datos de temperatura de una semana
temperaturas = [18, 19, 17, 20, 21, 19, 18]

temp_media = mean(temperaturas)      # 18.86°C
temp_max = max(temperaturas)         # 21°C
temp_std = std(temperaturas)         # 1.35°C (variabilidad)
temp_mediana = median(temperaturas)  # 19°C
```

**En PPT:**
```python
# Últimas 10 jugadas del oponente
ultimas_jugadas = ["piedra", "papel", "piedra", "piedra", "tijera", ...]

freq_piedra = count("piedra") / 10    # 0.4 (40%)
freq_papel = count("papel") / 10      # 0.3 (30%)
freq_tijera = count("tijera") / 10    # 0.3 (30%)
jugada_mas_comun = mode(ultimas_jugadas)  # "piedra"
```

### 3.4 Features Temporales

Capturan información sobre el **tiempo** o **secuencias**:

**Absolutas:**
```python
hora_del_dia = 14        # 2 PM
dia_semana = "lunes"
mes = "octubre"
trimestre = 4
```

**Relativas:**
```python
dias_desde_evento = 5
segundos_desde_inicio = 120
ronda_actual = 25
```

**Ventanas temporales:**
```python
# Últimos 5 elementos
ventana_5 = jugadas[-5:]

# Media móvil
media_movil_7 = mean(ultimas_7_jugadas)
```

**En PPT:**
```python
numero_ronda = 15                    # Posición en el juego
tiempo_reaccion = 2.3                # Segundos
racha_victorias_actual = 3           # Victorias consecutivas
jugadas_desde_cambio_patron = 5      # Jugadas desde que cambió
```

### 3.5 Features de Codificación (Encoding)

Convertir datos categóricos a numéricos:

#### A) Label Encoding
```python
# Categorías ordinales (con orden)
talla = "M"

talla_encoded:
  "XS" → 0
  "S"  → 1
  "M"  → 2  ← Nuestro valor
  "L"  → 3
  "XL" → 4
```

#### B) One-Hot Encoding
```python
# Categorías nominales (sin orden)
color = "rojo"

color_rojo   = 1  ← Activo
color_azul   = 0
color_verde  = 0

# Representación: [1, 0, 0]
```

**En PPT:**
```python
# Label encoding (hay orden implícito)
jugada = "papel"
jugada_encoded:
  "piedra" → 0
  "papel"  → 1  ← Nuestro valor
  "tijera" → 2

# One-hot encoding (sin orden)
jugada = "papel"
es_piedra = 0
es_papel  = 1  ← Activo
es_tijera = 0
```

**¿Cuándo usar cada uno?**
- **Label Encoding**: Cuando hay orden natural (tallas, niveles)
- **One-Hot Encoding**: Cuando no hay orden (colores, categorías)

### 3.6 Features de Interacción

Combinan múltiples features para capturar relaciones:

**Multiplicación:**
```python
# Original
area_casa = 100  # m²
precio_m2 = 2000  # €/m²

# Interacción
precio_total = area_casa * precio_m2  # 200,000€
```

**Comparación:**
```python
# Original
altura_jugador1 = 180  # cm
altura_jugador2 = 175  # cm

# Interacción
ventaja_altura = altura_jugador1 - altura_jugador2  # +5 cm
tiene_ventaja = (ventaja_altura > 0)  # True
```

**En PPT:**
```python
# Features individuales
ultima_jugada_mia = "piedra"
ultima_jugada_oponente = "papel"

# Feature de interacción
resultado_ultimo_enfrentamiento = comparar(ultima_jugada_mia, ultima_jugada_oponente)
# → "derrota"

oponente_gano_ultimo = (resultado_ultimo_enfrentamiento == "derrota")
# → True (el oponente puede ganar confianza)
```

### 3.7 Features de Dominio (Domain-Specific)

Basadas en conocimiento experto del problema:

**Ejemplo: Medicina**
```python
# Básico
peso = 80      # kg
altura = 1.75  # m

# Dominio médico
imc = peso / (altura ** 2)  # 26.1 (sobrepeso)
categoria_imc = "sobrepeso"
riesgo_cardiovascular = calcular_riesgo(edad, imc, presion)
```

**Ejemplo: Finanzas**
```python
# Básico
precio_accion = 50
precio_hace_1_mes = 45

# Dominio financiero
rendimiento = (precio_accion - precio_hace_1_mes) / precio_hace_1_mes
volatilidad = std(precios_ultimos_30_dias)
rsi = calcular_rsi(precios)  # Relative Strength Index
```

---

## 4. Técnicas de Feature Engineering

### 4.1 Extracción de Componentes

Descomponer información compleja en partes:

```python
# Fecha completa
fecha = "2024-10-30 14:30:25"

# Componentes extraídos
año = 2024
mes = 10
dia = 30
hora = 14
minuto = 30
segundo = 25
dia_semana = "miércoles"
es_fin_semana = False
trimestre = 4
estacion = "otoño"
```

### 4.2 Binning (Discretización)

Convertir variables continuas en categóricas:

```python
# Continua
edad = 25

# Binning por rangos
edad_grupo = binning(edad, bins=[0, 18, 30, 50, 100])
# → "18-30"

edad_categoria:
  0-18   → "menor"
  18-30  → "joven"   ← 25 cae aquí
  30-50  → "adulto"
  50+    → "mayor"
```

**Ventajas:**
- Reduce ruido
- Captura relaciones no lineales
- Más interpretable

**Desventajas:**
- Pierde información precisa
- Sensible a la elección de bins

**En PPT:**
```python
# Continuo
tiempo_reaccion = 2.3  # segundos

# Binning
velocidad_reaccion:
  < 1.0s  → "muy_rapido"
  1-2s    → "rapido"
  2-3s    → "normal"     ← 2.3 cae aquí
  3-5s    → "lento"
  > 5s    → "muy_lento"
```

### 4.3 Transformaciones Matemáticas

Cambiar la distribución de los datos:

#### Logaritmo
```python
# Para datos con crecimiento exponencial
ingresos_originales = [1000, 10000, 100000, 1000000]
log_ingresos = log(ingresos_originales)
# → [3, 4, 5, 6]  (escala más manejable)
```

**Cuándo usar:** Distribuciones muy sesgadas, valores que crecen exponencialmente

#### Raíz cuadrada
```python
# Para reducir el impacto de valores extremos
distancia = 100
sqrt_distancia = sqrt(100) = 10
```

**Cuándo usar:** Datos con valores atípicos moderados

#### Normalización (Min-Max)
```python
# Escalar a rango [0, 1]
valores = [10, 20, 30, 40, 50]

normalizado = (valor - min) / (max - min)
# 10 → 0.0
# 30 → 0.5
# 50 → 1.0
```

**Cuándo usar:** Cuando quieres todas las features en la misma escala

#### Estandarización (Z-score)
```python
# Centrar en 0 con desviación estándar 1
valores = [10, 20, 30, 40, 50]
media = 30
std = 14.14

z_score = (valor - media) / std
# 10 → -1.41
# 30 → 0
# 50 → +1.41
```

**Cuándo usar:** Para algoritmos sensibles a escala (SVM, redes neuronales)

### 4.4 Ventanas Deslizantes (Sliding Windows)

Capturar información de secuencias:

```python
# Secuencia de ventas diarias
ventas = [100, 120, 110, 130, 125, 140, 135]

# Ventana de 3 días
ventana_3 = ventas[-3:]  # [125, 140, 135]

# Features de la ventana
media_ultimos_3 = mean(ventana_3)      # 133.3
max_ultimos_3 = max(ventana_3)         # 140
tendencia = ventas[-1] - ventas[-3]     # 135 - 125 = +10
```

**En PPT:**
```python
# Historial de jugadas
jugadas = ["piedra", "papel", "piedra", "tijera", "papel"]

# Ventana de 3 jugadas
ventana_3 = jugadas[-3:]  # ["piedra", "tijera", "papel"]

# Features de ventana
patron_3 = "".join([j[0] for j in ventana_3])  # "ptp"
jugada_mas_frecuente_3 = mode(ventana_3)        # "piedra"
cambios_en_3 = count_changes(ventana_3)         # 3 (todas diferentes)
```

### 4.5 Lag Features (Retardos)

Valores pasados como features:

```python
# Serie temporal
dia    valor
1      100
2      105
3      110
4      108
5      115

# Con lag features
dia  valor  lag_1  lag_2  lag_3
1    100    NaN    NaN    NaN
2    105    100    NaN    NaN
3    110    105    100    NaN
4    108    110    105    100
5    115    108    110    105
```

**En PPT:**
```python
# Ronda actual: 5
ronda  jugada     lag_1     lag_2     lag_3
1      piedra     NaN       NaN       NaN
2      papel      piedra    NaN       NaN
3      piedra     papel     piedra    NaN
4      tijera     piedra    papel     piedra
5      papel      tijera    piedra    papel
```

### 4.6 Features de Frecuencia

Contar ocurrencias en diferentes ventanas:

```python
# Conteos globales
freq_total_piedra = count_total("piedra") / total_jugadas

# Conteos en ventanas
freq_ultimas_10_piedra = count_window("piedra", window=10) / 10
freq_ultimas_5_piedra = count_window("piedra", window=5) / 5

# Frecuencias relativas
ratio_freq_reciente_vs_global = freq_ultimas_5 / freq_total
# > 1.0 → jugando más piedra últimamente
# < 1.0 → jugando menos piedra últimamente
```

### 4.7 Features de Entropía

Medir la aleatoriedad o predictibilidad:

```python
# Fórmula de entropía de Shannon
H = -Σ P(x) * log2(P(x))

# Ejemplo 1: Completamente predecible
jugadas = ["piedra", "piedra", "piedra", "piedra"]
P(piedra) = 1.0
H = -1.0 * log2(1.0) = 0 bits  ← Sin incertidumbre

# Ejemplo 2: Completamente aleatorio
jugadas = ["piedra", "papel", "tijera", "piedra", "papel", "tijera"]
P(piedra) = P(papel) = P(tijera) = 0.33
H = -3 * (0.33 * log2(0.33)) ≈ 1.58 bits  ← Máxima incertidumbre

# Ejemplo 3: Parcialmente predecible
jugadas = ["piedra", "piedra", "piedra", "papel"]
P(piedra) = 0.75, P(papel) = 0.25
H = -(0.75*log2(0.75) + 0.25*log2(0.25)) ≈ 0.81 bits
```

**Interpretación:**
- H ≈ 0: Muy predecible (siempre la misma jugada)
- H ≈ 1.58: Muy aleatorio (distribución uniforme)
- H intermedio: Tiene patrones pero con variación

---

## 5. Aplicación al Proyecto PPT

### 5.1 Análisis del Problema

**Contexto:**
Queremos predecir la próxima jugada del oponente basándonos en el historial del juego.

**Desafíos:**
1. Los humanos no son completamente aleatorios (sesgo psicológico)
2. Los patrones cambian durante el juego (adaptación)
3. Datos secuenciales (orden importa)
4. Tenemos información mínima al inicio

**Información disponible (datos crudos del dataset):**

Tu CSV se ve así:
```csv
numero_ronda,jugada_jugador,jugada_oponente
1,piedra,papel
2,papel,piedra
3,piedra,tijera
...
```

**Eso es todo.** Solo tres columnas simples. Nada de tiempo de reacción, ni resultados precalculados, ni métricas complejas. Pero a partir de aquí podemos crear **más de 30 features útiles**.

**Ejemplo en Python:**
```python
ronda: 15
jugada_jugador: "piedra"
jugada_oponente: "papel"
```

**Primero: Calcular el resultado**
```python
def calcular_resultado(jugada_jugador, jugada_oponente):
    """
    Calcula el resultado desde la perspectiva del jugador
    """
    if jugada_jugador == jugada_oponente:
        return "empate"

    gana = {
        "piedra": "tijera",
        "papel": "piedra",
        "tijera": "papel"
    }

    if gana[jugada_jugador] == jugada_oponente:
        return "victoria"
    else:
        return "derrota"

# Ejemplo
resultado = calcular_resultado("piedra", "papel")  # → "derrota"
```

Con esta función, ya tenemos una cuarta columna derivada: `resultado`

### 5.2 Categorías de Features para PPT

#### Categoría 1: Features de Frecuencia

**Objetivo:** Detectar preferencias del oponente

```python
# Global (todo el juego)
freq_global_piedra = count_total("piedra") / total_jugadas
freq_global_papel = count_total("papel") / total_jugadas
freq_global_tijera = count_total("tijera") / total_jugadas

# Ventana reciente (últimas 10 jugadas)
freq_reciente_10_piedra = count_window("piedra", 10) / 10
freq_reciente_10_papel = count_window("papel", 10) / 10
freq_reciente_10_tijera = count_window("tijera", 10) / 10

# Ventana muy reciente (últimas 3 jugadas)
freq_reciente_3_piedra = count_window("piedra", 3) / 3
freq_reciente_3_papel = count_window("papel", 3) / 3
freq_reciente_3_tijera = count_window("tijera", 3) / 3

# Cambio de tendencia
cambio_tendencia_piedra = freq_reciente_3 - freq_global
# > 0 → jugando más piedra últimamente
# < 0 → jugando menos piedra últimamente
```

**Insight psicológico:**
- La gente tiende a tener jugadas favoritas
- Las preferencias cambian según el contexto

#### Categoría 2: Features de Patrones Secuenciales

**Objetivo:** Detectar secuencias repetitivas

```python
# Última jugada (lag-1)
ultima_jugada = jugadas[-1]

# Últimas 2 jugadas (bigrama)
patron_2 = (jugadas[-2], jugadas[-1])
# Ejemplo: ("piedra", "papel")

# Últimas 3 jugadas (trigrama)
patron_3 = (jugadas[-3], jugadas[-2], jugadas[-1])
# Ejemplo: ("tijera", "piedra", "papel")

# Frecuencia de patrones
freq_patron_2 = count_pattern(patron_2) / (total_jugadas - 1)
freq_patron_3 = count_pattern(patron_3) / (total_jugadas - 2)

# ¿Es un patrón nuevo o conocido?
es_patron_nuevo = (freq_patron_3 == 1)
```

**Codificación de patrones:**
```python
# One-hot encoding para bigrama
patron = ("piedra", "papel")

# Todas las combinaciones posibles: 3 × 3 = 9
patron_piedra_piedra = 0
patron_piedra_papel = 1   ← Activo
patron_piedra_tijera = 0
patron_papel_piedra = 0
patron_papel_papel = 0
# ... (9 features en total)
```

**Insight psicológico:**
- Los humanos caen en patrones repetitivos sin darse cuenta
- "Piedra, papel, piedra, papel..." es más común de lo esperado

#### Categoría 3: Features de Rachas

**Objetivo:** Capturar momentum y presión psicológica

```python
# Racha actual de victorias/derrotas
racha_victorias_actual = count_consecutive_wins()
racha_derrotas_actual = count_consecutive_losses()

# Tipo de racha
tipo_racha = "victoria" if racha_victorias_actual > 0 else "derrota"

# Longitud de racha (binned)
longitud_racha = max(racha_victorias_actual, racha_derrotas_actual)
racha_categoria:
  0      → "sin_racha"
  1-2    → "racha_corta"
  3-5    → "racha_media"
  6+     → "racha_larga"

# Rachas máximas en el juego
max_racha_victorias = max_consecutive_wins_ever()
max_racha_derrotas = max_consecutive_losses_ever()

# Proximidad a récord
cerca_record_victorias = (racha_victorias_actual >= max_racha_victorias - 1)
```

**Insight psicológico:**
- Después de perder varias veces, la gente tiende a cambiar
- Después de ganar, pueden ser más conservadores o agresivos

#### Categoría 4: Features de Respuesta al Resultado

**Objetivo:** Modelar cómo reacciona el oponente a victorias/derrotas

```python
# ¿Qué hizo después de su última victoria?
jugada_despues_victoria_anterior = get_jugada_after_last("victoria")

# ¿Qué hizo después de su última derrota?
jugada_despues_derrota_anterior = get_jugada_after_last("derrota")

# Frecuencias de reacción
freq_cambiar_despues_ganar = count_changes_after("victoria") / count("victoria")
freq_cambiar_despues_perder = count_changes_after("derrota") / count("derrota")

# Patrones de reacción
# Si perdió con papel, ¿qué juega?
jugada_comun_despues_perder_con = {
    "piedra": mode(jugadas_after_loss_with("piedra")),
    "papel": mode(jugadas_after_loss_with("papel")),
    "tijera": mode(jugadas_after_loss_with("tijera"))
}
```

**Insight psicológico:**
- "Aprendizaje por refuerzo": Si una jugada funcionó, repetir
- "Evitación": Si una jugada falló, evitarla
- Algunos jugadores hacen lo contrario (counter-thinking)

#### Categoría 5: Features Temporales (Fase del Juego)

**Objetivo:** Capturar cambios de estrategia según la fase del juego

```python
# Posición en el juego
progreso_juego = numero_ronda / total_rondas_estimadas
# Si no sabemos el total, podemos asumir 50 rondas típicas

fase_juego:
  0.0-0.33  → "inicio"    (rondas 1-17 si total=50)
  0.33-0.66 → "medio"     (rondas 18-33)
  0.66-1.0  → "final"     (rondas 34-50)

# One-hot encoding de la fase
fase_inicio = 1 if progreso_juego < 0.33 else 0
fase_medio = 1 if 0.33 <= progreso_juego < 0.66 else 0
fase_final = 1 if progreso_juego >= 0.66 else 0

# Número de ronda (valor numérico)
numero_ronda_normalizado = numero_ronda / 50  # Entre 0 y 1
```

**Insight psicológico:**
- **Inicio:** La gente explora y prueba estrategias diferentes
- **Medio:** Se establecen patrones más claros y estables
- **Final:** Puede haber cambios drásticos (presión por ganar, experimentación)

**Nota:** Si en el futuro añades tiempo de reacción al dataset, puedes agregar más features temporales como velocidad de decisión y cambios de ritmo.

#### Categoría 6: Features de Entropía y Aleatoriedad

**Objetivo:** Medir qué tan predecible es el oponente

```python
# Entropía global
entropia_global = calcular_entropia(todas_jugadas)
# 0 = muy predecible, 1.58 = completamente aleatorio

# Entropía en ventanas
entropia_ultimas_10 = calcular_entropia(jugadas[-10:])
entropia_ultimas_5 = calcular_entropia(jugadas[-5:])

# Cambio en entropía
esta_volviendose_mas_aleatorio = (entropia_ultimas_5 > entropia_global)

# Nivel de predictibilidad
nivel_predictibilidad:
  H < 0.5   → "muy_predecible"
  0.5-1.0   → "algo_predecible"
  1.0-1.4   → "poco_predecible"
  > 1.4     → "casi_aleatorio"
```

#### Categoría 7: Features de Counter-Strategy

**Objetivo:** Modelar si el oponente está adaptándose a nosotros

```python
# ¿Mis jugadas son predecibles?
mi_entropia = calcular_entropia(mis_jugadas)
soy_predecible = (mi_entropia < 1.0)

# ¿El oponente está explotando mis patrones?
mi_patron_mas_comun = mode(mis_ultimas_5_jugadas)
contador_de_mi_patron = get_counter(mi_patron_mas_comun)
oponente_usando_contador = (jugada_oponente_ultima == contador_de_mi_patron)

# Frecuencia de counter
freq_oponente_counter_a_mi = count_counters() / total_jugadas
oponente_esta_adaptandose = (freq_oponente_counter_a_mi > 0.4)
```

#### Categoría 8: Features de Transición (Markov)

**Objetivo:** Modelar probabilidades de transición

```python
# Matriz de transición: P(siguiente | actual)
transiciones = {
    ("piedra", "piedra"): count(["piedra", "piedra"]) / count_starting_with("piedra"),
    ("piedra", "papel"): count(["piedra", "papel"]) / count_starting_with("piedra"),
    ("piedra", "tijera"): count(["piedra", "tijera"]) / count_starting_with("piedra"),
    # ... 9 transiciones totales
}

# Jugada más probable según Markov
jugada_mas_probable_markov = max_transition_from(ultima_jugada)

# Confianza en la predicción
confianza_markov = transiciones[(ultima_jugada, jugada_mas_probable_markov)]
# Alta (>0.5) = oponente es muy predecible en transiciones
# Baja (<0.35) = transiciones casi aleatorias
```

### 5.3 Ejemplo Completo: De Datos Crudos a Features

**Situación:**
```python
# Dataset básico (lo que los alumnos tienen)
jugadas_jugador = [
    "papel", "piedra", "piedra", "papel", "piedra",
    "piedra", "tijera", "tijera", "papel", "tijera"
]

jugadas_oponente = [
    "piedra", "piedra", "papel", "piedra", "tijera",
    "piedra", "papel", "papel", "piedra", "papel"
]

# Ronda actual
numero_ronda = 10

# PASO 1: Calcular resultados (lo harán los alumnos)
resultados = []
for j_jug, j_op in zip(jugadas_jugador, jugadas_oponente):
    resultados.append(calcular_resultado(j_jug, j_op))

# resultados ahora contiene:
# ["victoria", "empate", "derrota", "victoria", "empate",
#  "empate", "victoria", "victoria", "empate", "victoria"]
```

**IMPORTANTE:** Los alumnos solo tienen 3 columnas en su CSV:
- `numero_ronda`, `jugada_jugador`, `jugada_oponente`

Todo lo demás (resultados, features) lo deben **calcular** ellos.

**Generación de features:**

```python
# ==== FRECUENCIAS ====
freq_global_piedra = 5/10 = 0.50   # 50% piedra
freq_global_papel = 4/10 = 0.40    # 40% papel
freq_global_tijera = 1/10 = 0.10   # 10% tijera

freq_ultimas_5_piedra = 2/5 = 0.40   # Bajando
freq_ultimas_5_papel = 3/5 = 0.60    # Subiendo
freq_ultimas_5_tijera = 0/5 = 0.00   # Sin tijera recientemente

cambio_tendencia_papel = 0.60 - 0.40 = +0.20  # Jugando más papel

# ==== PATRONES ====
ultima_jugada = "papel"
penultima_jugada = "piedra"
antepenultima_jugada = "papel"

bigrama_actual = ("piedra", "papel")
trigrama_actual = ("papel", "piedra", "papel")

# Este patrón apareció antes: papel → piedra → papel (posiciones 2,3,4 y 6,7,8)
freq_trigrama = 2/8 = 0.25

# ==== RACHAS ====
racha_actual = analizar_racha(resultados)
# ["victoria", "derrota", "victoria", "?"]
# Última fue victoria, racha = 1

racha_victorias_actual = 1
racha_derrotas_actual = 0
max_racha_derrotas = 2  # Las 2 derrotas consecutivas del medio

# ==== RESPUESTA A RESULTADOS ====
ultima_fue_victoria = True
jugada_despues_ultima_victoria = "papel"  # (posición 4 → 5)
# Después de victoria en pos 1: jugó piedra (pos 2)
# Después de victoria en pos 4: jugó tijera (pos 5)
# Patrón: varía después de ganar

# ==== TEMPORALES ====
progreso_juego = 10 / 50 = 0.20  # 20% del juego
fase_juego = "inicio"  # ronda 10 de esperadas 50
fase_inicio = 1
fase_medio = 0
fase_final = 0

# ==== ENTROPÍA ====
entropia_global = -[0.5*log(0.5) + 0.4*log(0.4) + 0.1*log(0.1)]
               ≈ 1.36  # Bastante aleatorio

entropia_ultimas_5 = -[0.4*log(0.4) + 0.6*log(0.6)]
                   ≈ 0.97  # Menos aleatorio recientemente

predictibilidad = "algo_predecible"

# ==== MARKOV ====
transiciones_piedra = {
    "piedra → piedra": 2/5 = 0.40,
    "piedra → papel": 2/5 = 0.40,
    "piedra → tijera": 1/5 = 0.20
}

ultima_jugada = "papel"
transiciones_papel = {
    "papel → piedra": 2/4 = 0.50,   ← Más probable
    "papel → papel": 1/4 = 0.25,
    "papel → tijera": 1/4 = 0.25
}

prediccion_markov = "piedra"  # con confianza 0.50
```

**Vector de features final:**

```python
features_vector = [
    # Frecuencias (6)
    0.50, 0.40, 0.10,  # global
    0.40, 0.60, 0.00,  # últimas 5

    # Cambios de tendencia (3)
    -0.10, +0.20, -0.10,  # piedra, papel, tijera

    # Lags (3) - one-hot de última jugada
    0, 1, 0,  # papel

    # Patrones (9) - one-hot de bigrama
    0, 0, 0,
    0, 0, 0,
    1, 0, 0,  # piedra→papel

    # Rachas (4)
    1,  # racha victorias actual
    0,  # racha derrotas actual
    2,  # max racha derrotas
    1,  # última fue victoria (bool)

    # Temporales (4) - solo fase del juego
    10,  # número de ronda
    0.20,  # progreso del juego
    1, 0, 0,  # fase: inicio (one-hot)

    # Entropía (2)
    1.36,  # global
    0.97,  # reciente

    # Markov (4)
    0.50, 0.40, 0.20,  # transiciones desde piedra
    0,  # predicción: piedra (one-hot)
]

# Total: 32 features (sin tiempo de reacción)
```

### 5.4 Código de Implementación

```python
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy

class PPTFeatureEngineering:
    """
    Clase para generar features del juego Piedra, Papel o Tijera
    """

    def __init__(self):
        self.jugadas_map = {"piedra": 0, "papel": 1, "tijera": 2}
        self.jugadas_reverse = {0: "piedra", 1: "papel", 2: "tijera"}

    def calcular_frecuencias(self, jugadas, ventana=None):
        """Calcula frecuencias de cada jugada"""
        if ventana:
            jugadas = jugadas[-ventana:]

        total = len(jugadas)
        if total == 0:
            return {"piedra": 0, "papel": 0, "tijera": 0}

        conteo = Counter(jugadas)
        return {
            "piedra": conteo.get("piedra", 0) / total,
            "papel": conteo.get("papel", 0) / total,
            "tijera": conteo.get("tijera", 0) / total
        }

    def crear_lag_features(self, jugadas, n_lags=3):
        """Crea features de jugadas anteriores"""
        features = {}
        for i in range(1, n_lags + 1):
            if len(jugadas) >= i:
                jugada = jugadas[-i]
                # One-hot encoding
                features[f'lag_{i}_piedra'] = 1 if jugada == "piedra" else 0
                features[f'lag_{i}_papel'] = 1 if jugada == "papel" else 0
                features[f'lag_{i}_tijera'] = 1 if jugada == "tijera" else 0
            else:
                features[f'lag_{i}_piedra'] = 0
                features[f'lag_{i}_papel'] = 0
                features[f'lag_{i}_tijera'] = 0
        return features

    def calcular_rachas(self, resultados):
        """Calcula rachas de victorias/derrotas"""
        if not resultados:
            return {"racha_victorias": 0, "racha_derrotas": 0}

        racha_actual = 0
        ultimo_resultado = resultados[-1]

        for resultado in reversed(resultados):
            if resultado == ultimo_resultado:
                racha_actual += 1
            else:
                break

        return {
            "racha_victorias": racha_actual if ultimo_resultado == "victoria" else 0,
            "racha_derrotas": racha_actual if ultimo_resultado == "derrota" else 0
        }

    def calcular_entropia(self, jugadas):
        """Calcula entropía de Shannon"""
        if not jugadas:
            return 0

        freq = self.calcular_frecuencias(jugadas)
        probs = [p for p in freq.values() if p > 0]
        return entropy(probs, base=2)

    def crear_patron_features(self, jugadas, n=2):
        """Crea features de patrones de n-gramas"""
        if len(jugadas) < n:
            return {}

        patron_actual = tuple(jugadas[-n:])

        # Contar cuántas veces apareció este patrón
        count = 0
        for i in range(len(jugadas) - n + 1):
            if tuple(jugadas[i:i+n]) == patron_actual:
                count += 1

        return {
            f'patron_{n}_frecuencia': count / max(1, len(jugadas) - n + 1)
        }

    def calcular_resultado(self, jugada_jugador, jugada_oponente):
        """Calcula el resultado desde la perspectiva del jugador"""
        if jugada_jugador == jugada_oponente:
            return "empate"

        gana = {
            "piedra": "tijera",
            "papel": "piedra",
            "tijera": "papel"
        }

        if gana[jugada_jugador] == jugada_oponente:
            return "victoria"
        else:
            return "derrota"

    def generar_features_completas(self, jugadas_oponente, jugadas_jugador,
                                   numero_ronda, total_rondas=50):
        """
        Genera el vector completo de features para una ronda

        Args:
            jugadas_oponente: lista de jugadas del oponente hasta ahora
            jugadas_jugador: lista de jugadas del jugador hasta ahora
            numero_ronda: número de ronda actual
            total_rondas: total estimado de rondas

        Returns:
            dict con todas las features
        """
        features = {}

        # Calcular resultados
        resultados = [self.calcular_resultado(j_jug, j_op)
                     for j_jug, j_op in zip(jugadas_jugador, jugadas_oponente)]

        # 1. Frecuencias
        freq_global = self.calcular_frecuencias(jugadas_oponente)
        features['freq_global_piedra'] = freq_global['piedra']
        features['freq_global_papel'] = freq_global['papel']
        features['freq_global_tijera'] = freq_global['tijera']

        freq_5 = self.calcular_frecuencias(jugadas_oponente, ventana=5)
        features['freq_5_piedra'] = freq_5['piedra']
        features['freq_5_papel'] = freq_5['papel']
        features['freq_5_tijera'] = freq_5['tijera']

        # 2. Cambios de tendencia
        features['cambio_tend_piedra'] = freq_5['piedra'] - freq_global['piedra']
        features['cambio_tend_papel'] = freq_5['papel'] - freq_global['papel']
        features['cambio_tend_tijera'] = freq_5['tijera'] - freq_global['tijera']

        # 3. Lags
        lag_features = self.crear_lag_features(jugadas_oponente, n_lags=3)
        features.update(lag_features)

        # 4. Rachas
        rachas = self.calcular_rachas(resultados)
        features.update(rachas)

        # 5. Temporales (solo fase del juego)
        features['numero_ronda'] = numero_ronda
        features['progreso_juego'] = numero_ronda / total_rondas
        progreso = features['progreso_juego']
        features['fase_inicio'] = 1 if progreso < 0.33 else 0
        features['fase_medio'] = 1 if 0.33 <= progreso < 0.66 else 0
        features['fase_final'] = 1 if progreso >= 0.66 else 0

        # 6. Entropía
        features['entropia_global'] = self.calcular_entropia(jugadas_oponente)
        features['entropia_5'] = self.calcular_entropia(jugadas_oponente[-5:])

        # 7. Patrones
        patron_2 = self.crear_patron_features(jugadas_oponente, n=2)
        patron_3 = self.crear_patron_features(jugadas_oponente, n=3)
        features.update(patron_2)
        features.update(patron_3)

        return features

# Ejemplo de uso
if __name__ == "__main__":
    fe = PPTFeatureEngineering()

    # Datos básicos (lo que tienen los alumnos)
    jugadas_jugador = ["papel", "piedra", "piedra", "papel", "piedra"]
    jugadas_oponente = ["piedra", "piedra", "papel", "piedra", "tijera"]

    features = fe.generar_features_completas(
        jugadas_oponente=jugadas_oponente,
        jugadas_jugador=jugadas_jugador,
        numero_ronda=5,
        total_rondas=50
    )

    print("Features generadas:")
    for nombre, valor in features.items():
        if isinstance(valor, float):
            print(f"{nombre:30s}: {valor:.3f}")
        else:
            print(f"{nombre:30s}: {valor}")
```

---

## 6. Validación y Selección de Features

### 6.1 ¿Cómo saber si una feature es buena?

#### A) Análisis de Correlación

Mide la relación lineal entre la feature y el objetivo:

```python
# Correlación de Pearson: [-1, +1]
correlacion = corr(feature, objetivo)

# Interpretación
|r| < 0.3   → Débil (puede no ser útil)
0.3 ≤ |r| < 0.7 → Moderada (probablemente útil)
|r| ≥ 0.7   → Fuerte (muy útil)
```

**Limitación:** Solo detecta relaciones lineales

#### B) Información Mutua

Mide dependencia no lineal:

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y)
# Valores más altos = feature más informativa
```

#### C) Feature Importance

Según el modelo entrenado:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
# Valores más altos = feature más importante
```

### 6.2 Selección de Features

#### A) Eliminación de Features Redundantes

Si dos features están muy correlacionadas (r > 0.9), eliminar una:

```python
# Matriz de correlación
corr_matrix = df.corr().abs()

# Encontrar pares con alta correlación
umbral = 0.9
features_redundantes = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > umbral:
            features_redundantes.add(corr_matrix.columns[i])
```

#### B) Recursive Feature Elimination (RFE)

Elimina iterativamente las features menos importantes:

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
selector.fit(X_train, y_train)

features_seleccionadas = X.columns[selector.support_]
```

#### C) Análisis Manual

```python
# Para cada feature, entrenar modelo con y sin ella
baseline_score = modelo.score(X_train, y_train)

for feature in features:
    X_sin_feature = X_train.drop(columns=[feature])
    score_sin = modelo.score(X_sin_feature, y_train)

    impacto = baseline_score - score_sin
    print(f"{feature}: impacto = {impacto:.4f}")

    # Si impacto ≈ 0, la feature no aporta
```

### 6.3 Validación Temporal (Critical para PPT)

**Importante:** En secuencias temporales, NO usar validación aleatoria

```python
# ❌ INCORRECTO para datos secuenciales
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, test_size=0.2)  # Mezcla tiempos

# ✅ CORRECTO para datos secuenciales
split_point = int(len(X) * 0.8)
X_train = X[:split_point]   # Primeras 80% de rondas
X_test = X[split_point:]    # Últimas 20% de rondas
```

**Razón:** El modelo no debe "ver el futuro" durante entrenamiento

---

## 7. Mejores Prácticas

### 7.1 Data Leakage (Filtración de Datos)

**Error crítico:** Incluir información del futuro en las features

```python
# ❌ MAL: Usa información que no tendríamos en predicción real
resultado_siguiente = resultados[i+1]  # No lo sabemos aún!

# ✅ BIEN: Solo usa información pasada
resultado_anterior = resultados[i-1]
```

### 7.2 Escalado de Features

Algoritmos sensibles a escala: SVM, KNN, Redes Neuronales

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Usar parámetros de train
```

**Importante:** Ajustar scaler solo con datos de entrenamiento

### 7.3 Manejo de Valores Faltantes

```python
# Opción 1: Eliminar filas (si son pocas)
df_clean = df.dropna()

# Opción 2: Imputar con media/mediana
df['feature'].fillna(df['feature'].mean(), inplace=True)

# Opción 3: Imputar con valor especial
df['feature'].fillna(-999, inplace=True)

# Opción 4: Crear feature indicadora
df['feature_faltante'] = df['feature'].isna().astype(int)
```

### 7.4 Feature Engineering Iterativo

1. Crear features iniciales
2. Entrenar modelo baseline
3. Analizar errores del modelo
4. Crear features para abordar errores
5. Repetir

### 7.5 Documentación

```python
# Documentar cada feature
features_doc = {
    'freq_global_piedra': 'Frecuencia de piedra en todo el juego',
    'racha_victorias': 'Número de victorias consecutivas actuales',
    'entropia_5': 'Entropía de últimas 5 jugadas (aleatoriedad)',
}
```

---

## 8. Ejercicios Propuestos

### Ejercicio 1: Frecuencias Básicas
Implementa una función que calcule:
- Frecuencia global de cada jugada
- Frecuencia en ventana de 10 jugadas
- Cambio de tendencia

```python
def calcular_frecuencias(jugadas):
    # Tu código aquí
    pass

# Test
jugadas = ["piedra"] * 7 + ["papel"] * 3
# Esperado: freq_piedra = 0.7, freq_papel = 0.3
```

### Ejercicio 2: Detección de Patrones
Encuentra todos los bigramas (pares de jugadas consecutivas) y sus frecuencias:

```python
def encontrar_bigramas(jugadas):
    # Tu código aquí
    pass

# Test
jugadas = ["piedra", "papel", "piedra", "tijera", "piedra", "papel"]
# Esperado: ("piedra", "papel") aparece 2 veces
```

### Ejercicio 3: Entropía
Implementa el cálculo de entropía de Shannon:

```python
def calcular_entropia(jugadas):
    # H = -Σ P(x) * log2(P(x))
    # Tu código aquí
    pass

# Test
jugadas1 = ["piedra"] * 10  # Esperado: H ≈ 0
jugadas2 = ["piedra", "papel", "tijera"] * 3  # Esperado: H ≈ 1.58
```

### Ejercicio 4: Features Completas
Integra todo en una función que genere un vector de features:

```python
def generar_features(historial, ronda_actual):
    """
    Args:
        historial: dict con jugadas, resultados, tiempos
        ronda_actual: int

    Returns:
        dict con todas las features
    """
    # Tu código aquí
    pass
```

### Ejercicio 5: Análisis de Importancia
Dado un dataset con features, entrena un modelo y determina cuáles son las 5 features más importantes:

```python
from sklearn.ensemble import RandomForestClassifier

def analizar_importancia(X, y):
    # Tu código aquí
    pass
```

### Ejercicio 6: Validación Temporal
Implementa una función de validación que respete el orden temporal:

```python
def temporal_train_test_split(X, y, test_size=0.2):
    """
    Split temporal (no aleatorio)
    """
    # Tu código aquí
    pass
```

---

## Recursos Adicionales

### Lecturas Recomendadas
1. "Feature Engineering for Machine Learning" - Alice Zheng & Amanda Casari
2. "The Art of Feature Engineering" - Pablo Duboue
3. Kaggle Feature Engineering Course: https://www.kaggle.com/learn/feature-engineering

### Herramientas Útiles
```python
# Feature engineering automatizado
import featuretools

# Selección de features
from sklearn.feature_selection import SelectKBest, RFE

# Visualización de importancia
import shap
import eli5
```

### Competiciones de Kaggle para Practicar
- Titanic (features básicas)
- House Prices (features numéricas)
- Categorical Feature Encoding Challenge

---

## Conclusión

**Feature Engineering es tanto arte como ciencia:**

- **Ciencia:** Métodos sistemáticos, métricas, validación
- **Arte:** Intuición del dominio, creatividad, experimentación

**Para el proyecto PPT:**
1. Empieza simple (frecuencias, lags)
2. Añade complejidad gradualmente (patrones, rachas)
3. Valida siempre temporalmente
4. Documenta qué funciona y qué no

**Recuerda:**
> "Better features beat fancier models" - Prof. Andrew Ng

La calidad de tus features determina el límite superior del rendimiento de tu IA.

---

**Última actualización:** Octubre 2024

**Autor:** Curso IA-CC-2025
