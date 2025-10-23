# Fundamentos de Inteligencia Artificial y Machine Learning
## Conceptos Teóricos y Matemáticos Fundamentales

---

# PARTE 1: INTRODUCCIÓN A LA INTELIGENCIA ARTIFICIAL

## 1.1 ¿Qué es la Inteligencia Artificial?

La **Inteligencia Artificial (IA)** es la simulación de procesos de inteligencia humana por parte de máquinas, especialmente sistemas informáticos.

### Niveles de IA:
1. **IA Débil (Narrow AI)**: Diseñada para tareas específicas
   - Ejemplo: Detectar spam, reconocer caras, recomendar productos, diagnóstico médico asistido
   - Es el tipo de IA que usamos actualmente en aplicaciones reales
   - Excelente en su dominio específico, pero no generalizable

2. **IA Fuerte (General AI)**: Capacidad humana generalizada
   - Todavía no existe (investigación en progreso)
   - Podría aprender y razonar en cualquier dominio como un humano

3. **Super IA**: Supera la inteligencia humana en todos los aspectos
   - Teórica/futura
   - Implicaciones éticas y filosóficas significativas

### Componentes clave de un sistema de IA:
```
ENTRADA (Datos) → PROCESAMIENTO (Algoritmo) → SALIDA (Predicción/Decisión)
     ↑                                              ↓
     └──────── RETROALIMENTACIÓN (Aprendizaje) ────┘
```

## 1.2 ¿Qué es Machine Learning?

**Machine Learning (ML)** es un subconjunto de la IA que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia **sin ser programados explícitamente**.

### Diferencia clave:
- **Programación tradicional**: 
  ```
  Reglas + Datos → Respuesta
  ```
- **Machine Learning**: 
  ```
  Datos + Respuestas → Reglas (Modelo)
  ```

### Ejemplos prácticos:
- **Tradicional**: "Si temperatura > 30°C, entonces activar aire acondicionado"
- **ML**: "Basándome en temperatura, humedad, hora del día y preferencias históricas, predecir configuración óptima del climatizador"

**Otro ejemplo:**
- **Tradicional**: "Si email contiene palabra 'viagra', entonces es spam"
- **ML**: "Basándome en millones de emails etiquetados, aprender patrones complejos que identifican spam"

---

# PARTE 2: FUNDAMENTOS MATEMÁTICOS

## 2.1 Probabilidad Básica

### Conceptos esenciales:

**Probabilidad**: Medida de la posibilidad de que ocurra un evento
- Rango: [0, 1] donde 0 = imposible, 1 = seguro
- P(A) = Casos favorables / Casos totales

**Ejemplos en Machine Learning:**
- **Clasificación de imágenes**: P(gato|imagen) = probabilidad de que la imagen contenga un gato
- **Detección de fraude**: P(fraude|transacción) = probabilidad de que una transacción sea fraudulenta
- **Predicción del clima**: P(lluvia|condiciones_actuales) = probabilidad de lluvia dadas las condiciones
- **Diagnóstico médico**: P(enfermedad|síntomas) = probabilidad de enfermedad dados los síntomas observados

### Regla de la Suma:
Para eventos mutuamente excluyentes:
```
P(A o B) = P(A) + P(B)

Ejemplo: Lanzar un dado
P(sacar 1 o 6) = P(1) + P(6) = 1/6 + 1/6 = 2/6 = 1/3

Aplicación ML: Clasificación multiclase
P(clase_A o clase_B) = P(clase_A) + P(clase_B)
```

### Regla del Producto:
Para eventos independientes:
```
P(A y B) = P(A) × P(B)

Ejemplo: Lanzar dos monedas
P(cara en moneda 1 Y cara en moneda 2) = 1/2 × 1/2 = 1/4

Aplicación ML: Features independientes
P(email_spam Y contiene_enlace) = P(email_spam) × P(contiene_enlace)
(solo si son independientes)
```

## 2.2 Probabilidad Condicional

**Definición**: Probabilidad de A dado que B ya ocurrió
```
P(A|B) = P(A ∩ B) / P(B)
```

**Ejemplos en Machine Learning:**
- **Filtro de spam**: P(spam | contiene_"oferta") = Probabilidad de que sea spam dado que contiene "oferta"
- **Reconocimiento facial**: P(persona_X | detecta_rostro) = Probabilidad de que sea la persona X dado que se detectó un rostro
- **Predicción de compra**: P(compra | visitó_3_veces) = Probabilidad de compra dado que visitó el sitio 3 veces
- **Diagnóstico**: P(diabetes | glucosa_alta) = Probabilidad de diabetes dado nivel alto de glucosa

### Teorema de Bayes:
```
P(A|B) = [P(B|A) × P(A)] / P(B)

Donde:
- P(A|B) = Probabilidad posterior (lo que queremos saber)
- P(B|A) = Verosimilitud (likelihood)
- P(A) = Probabilidad a priori
- P(B) = Evidencia (marginal)
```

**Aplicación práctica - Detección de spam:**
Queremos calcular: P(spam | contiene_"gratis")

Datos conocidos:
- P(contiene_"gratis" | spam) = 0.8 (80% de spams contienen "gratis")
- P(spam) = 0.3 (30% de emails son spam a priori)
- P(contiene_"gratis") = 0.4 (40% de todos los emails contienen "gratis")

Aplicando Bayes:
P(spam | contiene_"gratis") = [0.8 × 0.3] / 0.4 = 0.6 = 60%

**Otro ejemplo - Diagnóstico médico:**
P(enfermedad | test_positivo) = [P(test_positivo | enfermedad) × P(enfermedad)] / P(test_positivo)

Esto es fundamental en clasificadores Naive Bayes.

## 2.3 Distribuciones de Probabilidad

### Distribución Uniforme:
Todos los resultados tienen igual probabilidad
- **Ejemplo**: Lanzar un dado justo: P(1) = P(2) = ... = P(6) = 1/6
- **En ML**: Inicialización de pesos en redes neuronales
- Entropía máxima (máxima incertidumbre)
- Sin sesgo hacia ninguna opción

### Distribución Normal (Gaussiana):
La distribución más común en naturaleza y datos
```
μ = media, σ = desviación estándar
f(x) = (1/(σ√2π)) × e^(-(x-μ)²/2σ²)
```
- **Ejemplo**: Altura de personas, errores de medición, muchas features en datasets
- **En ML**: Asumida en muchos algoritmos (regresión lineal, LDA)
- Propiedad: 68% de datos dentro de ±1σ, 95% dentro de ±2σ

### Distribución Binomial:
Probabilidad de k éxitos en n intentos
- **Ejemplo**: Número de caras en 10 lanzamientos de moneda
- **En ML**: Clasificación binaria, A/B testing

### Distribución Sesgada (Skewed):
Algunas opciones más probables que otras
- **Ejemplo**: Distribución de ingresos (pocos muy ricos, muchos con ingresos medios/bajos)
- **En ML**: Datasets desbalanceados (fraude: 99% legítimo, 1% fraude)
- Entropía menor → más predecible
- Requiere técnicas especiales de balance

## 2.4 Valor Esperado

**Definición**: Promedio ponderado de todos los resultados posibles
```
E[X] = Σ (valor_i × probabilidad_i)
```

**Ejemplo 1: Inversión**
Invertir $1000 con posibles resultados:
- 30% probabilidad de ganar $500 → valor = $1500
- 50% probabilidad de ganar $100 → valor = $1100
- 20% probabilidad de perder $200 → valor = $800

E[inversión] = (1500 × 0.3) + (1100 × 0.5) + (800 × 0.2) = $1160
Ganancia esperada = $1160 - $1000 = $160

**Ejemplo 2: Clasificador ML**
Resultados posibles:
- Correcto (ganancia = +10), P = 0.8
- Incorrecto (costo = -30), P = 0.2

E[decisión] = (10 × 0.8) + (-30 × 0.2) = 8 - 6 = +2 (beneficio neto positivo)

**Aplicación en ML:**
- Decisiones bajo incertidumbre
- Optimización de funciones de costo
- Evaluación de estrategias de negocio con modelos predictivos

## 2.5 Entropía de Shannon

**Definición**: Medida del desorden o incertidumbre en un conjunto de datos
```
H(X) = -Σ P(xi) × log₂(P(xi))
```

**Interpretación:**
- H = 0: Certeza total (sin incertidumbre)
- H = log₂(n): Máxima entropía para n opciones equiprobables
- H intermedia: Incertidumbre parcial

**Ejemplo 1: Lanzamiento de moneda**
Moneda justa:
- P(cara) = 0.5, P(cruz) = 0.5
- H = -(0.5×log₂(0.5) + 0.5×log₂(0.5)) = 1 bit
- Máxima incertidumbre para 2 opciones

Moneda trucada:
- P(cara) = 0.9, P(cruz) = 0.1
- H = -(0.9×log₂(0.9) + 0.1×log₂(0.1)) ≈ 0.47 bits
- Más predecible, menor incertidumbre

**Ejemplo 2: Clasificación de emails**
Dataset balanceado:
- P(spam) = 0.5, P(no_spam) = 0.5
- H = 1 bit (máxima incertidumbre)

Dataset desbalanceado:
- P(spam) = 0.05, P(no_spam) = 0.95
- H ≈ 0.29 bits (muy predecible: casi siempre no-spam)

**Aplicación en ML:**
- **Árboles de decisión**: Maximizar ganancia de información (reducir entropía)
- **Feature engineering**: Features con alta entropía → más información
- **Evaluación de modelos**: Entropía cruzada (cross-entropy) como función de pérdida
- **Compresión de datos**: Mayor entropía → más difícil comprimir

---

# PARTE 3: TIPOS DE APRENDIZAJE AUTOMÁTICO

## 3.1 Aprendizaje Supervisado

**Definición**: Aprendemos de ejemplos etiquetados (sabemos la respuesta correcta)

**Características:**
- Tenemos pares (entrada, salida esperada)
- El modelo aprende la relación entrada→salida
- Objetivo: Predecir salidas para nuevas entradas

**Ejemplos reales:**

**1. Clasificación de emails:**
```
Entrenamiento:
email1: "Compra ahora descuento..." → spam
email2: "Reunión mañana a las 10..." → no-spam
email3: "Gana dinero rápido..." → spam
...
Predicción:
email_nuevo: "Oferta exclusiva..." → ??? (el modelo predice: spam)
```

**2. Reconocimiento de dígitos escritos a mano:**
```
Entrenamiento:
imagen1: [pixels...] → 7
imagen2: [pixels...] → 3
imagen3: [pixels...] → 7
...
Predicción:
imagen_nueva: [pixels...] → ??? (el modelo predice: 7)
```

**3. Predicción de precio de casas:**
```
Entrenamiento:
casa1: (100m², 2 hab, centro) → $200,000
casa2: (80m², 1 hab, periferia) → $120,000
casa3: (150m², 3 hab, centro) → $350,000
...
Predicción:
casa_nueva: (120m², 2 hab, centro) → ??? (el modelo predice: $280,000)
```

### Algoritmos comunes:

**Regresión** (predecir valores continuos):
- **Regresión Lineal**: y = mx + b (ejemplo: predecir precio de casa)
- **Regresión Polinómica**: relaciones no lineales
- **Random Forest Regressor**: combinación de árboles
- Ejemplos de uso: Predecir ventas, temperatura, precios de acciones

**Clasificación** (predecir categorías discretas):
- **Regresión Logística**: clasificación binaria (sí/no, spam/no-spam)
- **Árboles de Decisión**: reglas if-then jerárquicas
- **Random Forest**: ensemble de árboles
- **Naive Bayes**: basado en probabilidades condicionales
- **Support Vector Machines (SVM)**: encuentra hiperplano óptimo
- **K-Nearest Neighbors (KNN)**: clasifica por vecinos más cercanos
- **Redes Neuronales**: modelos complejos inspirados en el cerebro
- Ejemplos de uso: Filtro spam, reconocimiento de imágenes, diagnóstico médico

## 3.2 Aprendizaje No Supervisado

**Definición**: Encontrar patrones en datos sin etiquetas

**Características:**
- Solo tenemos entradas, no salidas
- El modelo encuentra estructura oculta
- Objetivo: Descubrir agrupaciones o patrones

**Ejemplos reales:**

**1. Segmentación de clientes (Clustering):**
```
Datos: [(edad, ingresos, compras_mes), ...]
       [(25, 30K, 5), (45, 80K, 15), (28, 35K, 6), ...]

Descubrimiento automático:
- Cluster 1: Jóvenes, ingresos medios, compras moderadas
- Cluster 2: Adultos, altos ingresos, muchas compras
- Cluster 3: Seniors, ingresos variables, pocas compras

Sin que le digamos qué grupos buscar!
```

**2. Detección de anomalías en transacciones:**
```
Datos: Miles de transacciones bancarias normales
Modelo aprende: Qué es "normal"
Resultado: Detecta transacciones sospechosas que se desvían del patrón
```

**3. Reducción de dimensionalidad:**
```
Datos originales: 1000 features por imagen
Después de PCA: 50 features principales que capturan 95% de la varianza
Beneficio: Procesamiento más rápido, menos ruido, visualización posible
```

### Técnicas principales:

**Clustering (agrupación):**
- **K-Means**: Agrupa datos en K clusters
- **DBSCAN**: Encuentra clusters de formas arbitrarias
- **Hierarchical Clustering**: Crea jerarquía de clusters
- Aplicaciones: Segmentación de mercado, organización de documentos, compresión de imágenes

**Reducción de Dimensionalidad:**
- **PCA (Principal Component Analysis)**: Encuentra direcciones de máxima varianza
- **t-SNE**: Visualización de datos de alta dimensión
- **Autoencoders**: Redes neuronales que aprenden representaciones compactas
- Aplicaciones: Visualización, preprocesamiento, compresión

**Detección de Anomalías:**
- **Isolation Forest**: Aísla observaciones anómalas
- **One-Class SVM**: Define región de normalidad
- **Autoencoders**: Detecta datos que reconstruye mal
- Aplicaciones: Fraude, fallos en sistemas, ciber-seguridad

**Reglas de Asociación:**
- **Apriori**: Encuentra patrones frecuentes
- **FP-Growth**: Versión optimizada
- Aplicación clásica: "Los clientes que compran X también compran Y" (market basket analysis)

## 3.3 Aprendizaje por Refuerzo

**Definición**: Aprender mediante prueba y error con recompensas/castigos

**Características:**
- Agente interactúa con entorno
- Recibe recompensas (+) o penalizaciones (-)
- Objetivo: Maximizar recompensa total

**Ejemplos reales:**

**1. Robot aprendiendo a caminar:**
```
Estado: Posiciones de articulaciones, sensores de equilibrio
Acción: Mover motor X con fuerza Y
Resultado: Robot avanza 10cm → Recompensa +10
         Robot se cae → Recompensa -100
Aprendizaje: Ajusta acciones para maximizar distancia recorrida sin caerse
```

**2. AlphaGo jugando Go:**
```
Estado: Configuración actual del tablero
Acción: Colocar ficha en posición (x,y)
Resultado: Ganar partida → Recompensa +1
          Perder partida → Recompensa -1
Aprendizaje: Aprende estrategias óptimas después de millones de partidas
```

**3. Coche autónomo:**
```
Estado: Velocidad, posición en carril, vehículos cercanos, señales
Acción: Acelerar, frenar, girar
Resultado: Mantiene carril y velocidad → Recompensa +1
          Salida de carril → Recompensa -10
          Accidente → Recompensa -1000
Aprendizaje: Política de conducción segura y eficiente
```

**4. Optimización de recursos en Data Center:**
```
Estado: Temperatura, carga de servidores, demanda
Acción: Ajustar enfriamiento, distribuir carga
Resultado: Ahorro de energía → Recompensa positiva
          Sobrecalentamiento → Recompensa negativa
Aprendizaje: Google logró reducir 40% del consumo energético en enfriamiento
```

### Componentes clave:

- **Agente**: El sistema que aprende y toma decisiones
- **Entorno**: El mundo con el que interactúa el agente
- **Estado (s)**: Situación actual del entorno
- **Acción (a)**: Decisión que toma el agente
- **Recompensa (r)**: Feedback inmediato por la acción
- **Política (π)**: Estrategia que mapea estados → acciones
- **Función de valor Q(s,a)**: Recompensa esperada a largo plazo

### Algoritmos principales:

- **Q-Learning**: Aprende valores Q de forma iterativa (tabular)
- **Deep Q-Network (DQN)**: Q-Learning con redes neuronales profundas
- **Policy Gradient**: Optimiza directamente la política
- **Actor-Critic**: Combina value-based y policy-based
- **A3C**: Asynchronous Advantage Actor-Critic
- **PPO**: Proximal Policy Optimization (usado en ChatGPT con RLHF)

## 3.4 ¿Cuándo usar cada tipo de aprendizaje?

### Decision Tree para elegir tipo de ML:

```
¿Tienes datos etiquetados?
│
├─ SÍ → ¿Quieres predecir algo?
│       │
│       ├─ SÍ → APRENDIZAJE SUPERVISADO
│       │       ├─ Salida continua (precio, temperatura) → REGRESIÓN
│       │       └─ Salida categórica (spam/no-spam) → CLASIFICACIÓN
│       │
│       └─ NO → ¿Solo quieres encontrar estructura?
│               └─ APRENDIZAJE NO SUPERVISADO
│
└─ NO → ¿Puedes definir recompensas/castigos?
        │
        ├─ SÍ → ¿El sistema interactúa con un entorno?
        │       └─ SÍ → APRENDIZAJE POR REFUERZO
        │
        └─ NO → APRENDIZAJE NO SUPERVISADO
                ├─ Agrupar datos similares → CLUSTERING
                ├─ Detectar datos raros → DETECCIÓN DE ANOMALÍAS
                └─ Reducir complejidad → REDUCCIÓN DE DIMENSIONALIDAD
```

### Tabla comparativa:

| Característica | Supervisado | No Supervisado | Refuerzo |
|----------------|-------------|----------------|----------|
| **Datos etiquetados** | Sí, necesarios | No necesarios | Opcional |
| **Objetivo** | Predecir salida | Encontrar estructura | Maximizar recompensa |
| **Feedback** | Etiquetas correctas | Ninguno | Recompensas |
| **Interacción** | Pasiva (datos fijos) | Pasiva | Activa (con entorno) |
| **Ejemplos** | Spam filter, precio casas | Segmentación, clustering | Robótica, juegos |
| **Complejidad** | Media | Media-Baja | Alta |
| **Datos necesarios** | Muchos (con etiquetas) | Muchos (sin etiquetas) | Muchas iteraciones |

### Tendencias actuales:

- **Self-Supervised Learning**: Híbrido entre supervisado y no supervisado (usado en GPT, BERT)
- **Semi-Supervised Learning**: Pocos datos etiquetados + muchos sin etiquetar
- **Transfer Learning**: Usar modelo pre-entrenado y adaptar a tu problema
- **Few-Shot Learning**: Aprender de muy pocos ejemplos
- **Meta-Learning**: "Aprender a aprender"

---

# PARTE 4: ALGORITMOS FUNDAMENTALES DE MACHINE LEARNING

## 4.1 Regresión Lineal

**Principio**: Encontrar la línea que mejor se ajusta a los datos

**Modelo matemático:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Donde:
- y = variable objetivo (lo que queremos predecir)
- x₁, x₂, ..., xₙ = features (variables independientes)
- β₀ = intercepto (bias)
- β₁, β₂, ..., βₙ = coeficientes (pesos)
- ε = error
```

**Función de costo (MSE - Mean Squared Error):**
```
J(β) = (1/n) Σ(y_real - y_pred)²
```

**Ventajas:**
- Simple y muy interpretable
- Rápido de entrenar
- Funciona bien con relaciones lineales
- Baseline excelente para comparar

**Desventajas:**
- Asume linealidad (puede no capturar relaciones complejas)
- Sensible a outliers
- Puede sufrir de multicolinealidad

**Aplicaciones:**
- Predicción de precios (casas, acciones)
- Estimación de ventas
- Análisis de tendencias
- Econometría

## 4.2 Cadenas de Markov

**Principio**: "El siguiente estado depende solo del estado actual" (Propiedad de Markov)

**Definición formal:**
```
P(X_{t+1} = s_{j} | X_t = s_i, X_{t-1} = s_{k}, ...) = P(X_{t+1} = s_j | X_t = s_i)
```

El futuro es independiente del pasado dado el presente.

**Matriz de transición P:**
```
        Estado_1  Estado_2  Estado_3
Estado_1   0.7      0.2       0.1
Estado_2   0.3      0.5       0.2
Estado_3   0.4      0.1       0.5
```

**Ejemplo: Predicción del clima**
```
Estados: {Soleado, Nublado, Lluvioso}

Matriz de transición:
              Soleado  Nublado  Lluvioso
Soleado        0.8      0.15     0.05
Nublado        0.3      0.4      0.3
Lluvioso       0.2      0.3      0.5

Interpretación:
- Si hoy está Soleado, mañana hay 80% de probabilidad de sol
- Si hoy está Lluvioso, mañana hay 50% de probabilidad de lluvia
```

**Predicción:**
```
Estado actual: Nublado
Probabilidades para mañana:
  P(Soleado) = 0.3
  P(Nublado) = 0.4
  P(Lluvioso) = 0.3

Predicción: Nublado (máxima probabilidad)
```

**Aplicaciones en ML:**
- Modelado de secuencias (texto, ADN, comportamiento de usuarios)
- Sistemas de recomendación
- Procesamiento de lenguaje natural (n-gramas)
- Reconocimiento de voz
- PageRank de Google (variante: Markov con random walk)

## 4.3 N-gramas y Modelos de Lenguaje

**Principio**: Predecir el siguiente elemento basándose en N elementos anteriores

**N-grama**: Secuencia de N elementos consecutivos (palabras, caracteres, etc.)

### Tipos de N-gramas:

**Unigrama (N=1)**: Elementos individuales
```
Texto: "El gato come pescado"
Unigramas: ["El", "gato", "come", "pescado"]
P("gato") = frecuencia("gato") / total_palabras
```

**Bigrama (N=2)**: Pares de elementos consecutivos
```
Texto: "El gato come pescado. El perro come carne."
Bigramas: [("El", "gato"), ("gato", "come"), ("come", "pescado"),
           ("El", "perro"), ("perro", "come"), ("come", "carne")]

Tabla de probabilidades condicionales:
P("gato" | "El") = 1/2 = 0.5
P("perro" | "El") = 1/2 = 0.5
P("pescado" | "come") = 1/2 = 0.5
P("carne" | "come") = 1/2 = 0.5
```

**Trigrama (N=3)**: Tripletes
```
Trigramas: [("El", "gato", "come"), ("gato", "come", "pescado"), ...]
P("come" | "El", "gato") = frecuencia("El gato come") / frecuencia("El gato")
```

### Ejemplo práctico: Autocompletado

```
Modelo entrenado con millones de textos:
Usuario escribe: "Buenos"
  → Predicciones bigramas: ["días" (60%), "noches" (20%), "Aires" (15%), ...]

Usuario escribe: "Buenos días"
  → Predicciones trigramas: ["." (40%), "señor" (20%), "a" (15%), ...]
```

### Aplicaciones:

**1. Procesamiento de Lenguaje Natural:**
- Corrección ortográfica y gramatical
- Autocompletado de texto (teclados, buscadores)
- Traducción automática
- Generación de texto

**2. Bioinformática:**
- Análisis de secuencias de ADN/proteínas
- Predicción de estructura genética

**3. Sistemas de recomendación:**
- "Los usuarios que vieron A y B, vieron C"
- Secuencias de compras

**4. Detección de anomalías:**
- Identificar patrones inusuales en logs de sistema
- Seguridad: detectar comportamientos anómalos

### Problema del N-grama: Sparsity

A mayor N, más precisión PERO:
- Necesitas MUCHO más datos
- Muchas combinaciones nunca vistas en entrenamiento
- Más memoria requerida

**Solución moderna**: Redes neuronales (LSTM, Transformers) que capturan contexto sin enumerar todas las combinaciones

## 4.4 K-Nearest Neighbors (KNN)

**Principio**: "Dime con quién andas y te diré quién eres" - Los puntos similares están cerca

**Algoritmo:**
1. Calcular distancia entre nuevo punto y todos los puntos de entrenamiento
2. Seleccionar los K vecinos más cercanos
3. Para clasificación: votar por la clase mayoritaria
4. Para regresión: promediar los valores

**Ejemplo: Clasificar una flor**
```
Nuevaflor: longitud_pétalo=5.1, ancho_pétalo=3.2

K=3 vecinos más cercanos:
  1. Iris-setosa (distancia=0.8)
  2. Iris-setosa (distancia=1.1)
  3. Iris-versicolor (distancia=1.5)

Predicción: Iris-setosa (2 votos vs 1)
```

**Distancias comunes:**

**Euclidiana** (más común):
```
d = √[(x₁-x₂)² + (y₁-y₂)²]
```

**Manhattan**:
```
d = |x₁-x₂| + |y₁-y₂|
```

**Minkowski** (generalización):
```
d = (|x₁-x₂|ᵖ + |y₁-y₂|ᵖ)^(1/p)
donde p=1 → Manhattan, p=2 → Euclidiana
```

**Elección de K:**
- K pequeño (1-3): Sensible a ruido, fronteras complejas
- K grande (>10): Más robusto pero fronteras muy suaves
- Regla empírica: K = √n donde n = número de muestras
- Usar K impar para evitar empates en clasificación binaria

**Ventajas:**
- Simple de entender e implementar
- No requiere entrenamiento (lazy learning)
- Funciona bien con fronteras no lineales
- Útil para datasets pequeños

**Desventajas:**
- Lento en predicción (debe calcular todas las distancias)
- Sensible a la escala de features (necesita normalización)
- Maldición de la dimensionalidad (alta dimensión → todo está "lejos")
- Requiere mucha memoria (guarda todos los datos)

**Aplicaciones:**
- Sistemas de recomendación ("usuarios similares a ti")
- Reconocimiento de patrones
- Clasificación de imágenes
- Detección de anomalías

## 4.5 Ensemble Methods (Métodos de Conjunto)

**Principio**: "Muchas cabezas piensan mejor que una"

Combinar múltiples modelos para obtener mejor rendimiento que cualquier modelo individual.

### Tipos de Ensemble:

**1. Voting (Votación)**

**Hard Voting** (mayoría simple):
```
Clasificador 1: Clase A
Clasificador 2: Clase A
Clasificador 3: Clase B
Clasificador 4: Clase A

Predicción final: Clase A (3 votos vs 1)
```

**Soft Voting** (promedio de probabilidades):
```
Clasificador 1: P(A)=0.6, P(B)=0.4
Clasificador 2: P(A)=0.8, P(B)=0.2
Clasificador 3: P(A)=0.5, P(B)=0.5

Promedio: P(A)=0.63, P(B)=0.37
Predicción final: Clase A
```

**2. Bagging (Bootstrap Aggregating)**

Ejemplo: **Random Forest**
```
Dataset original: 1000 muestras

1. Crear N subsets con reemplazo (bootstrap)
   Subset1: muestra aleatoria de 1000 (con repeticiones)
   Subset2: muestra aleatoria de 1000
   ...
   Subset100: muestra aleatoria de 1000

2. Entrenar un árbol de decisión en cada subset

3. Predicción final: votar (clasificación) o promediar (regresión)
```

**Ventajas:**
- Reduce varianza (overfitting)
- Funciona bien con modelos inestables (árboles)
- Paralelizable

**3. Boosting (Impulso)**

Modelos secuenciales donde cada uno corrige errores del anterior.

**AdaBoost** (Adaptive Boosting):
```
1. Entrenar modelo1 con datos
2. Identificar muestras mal clasificadas
3. Aumentar peso de muestras difíciles
4. Entrenar modelo2 enfocándose en errores de modelo1
5. Repetir N veces
6. Predicción final: voto ponderado de todos los modelos
```

**Gradient Boosting** (más popular: XGBoost, LightGBM, CatBoost):
```
1. Modelo inicial: predicción simple (media)
2. Calcular residuos (errores)
3. Entrenar modelo para predecir residuos
4. Actualizar predicción: predicción_anterior + learning_rate × modelo_residuos
5. Repetir hasta convergencia
```

**Ventajas:**
- Reduce bias (underfitting)
- Muy potente en competiciones (Kaggle)
- Captura patrones complejos

**Desventajas:**
- Más lento de entrenar (secuencial)
- Puede hacer overfitting si no se regula

**4. Stacking (Apilamiento)**

```
Nivel 1 (Base Models):
  - Modelo A: KNN
  - Modelo B: SVM
  - Modelo C: Árbol de Decisión

Nivel 2 (Meta-Model):
  Entrada: predicciones de A, B, C
  Modelo: Regresión Logística
  Salida: Predicción final
```

### Ejemplo práctico:

**Problema**: Predecir si un email es spam

```
Random Forest (100 árboles): Accuracy = 92%
Gradient Boosting: Accuracy = 93%
SVM: Accuracy = 91%

Ensemble (Voting): Accuracy = 95%
```

**¿Por qué funciona?**
- Modelos diferentes cometen errores diferentes
- Al combinarlos, los errores se cancelan
- La "sabiduría de las masas"

### Cuándo usar cada tipo:

- **Bagging**: Datos con mucho ruido, reducir varianza
- **Boosting**: Mejorar modelos débiles, capturar patrones complejos
- **Stacking**: Maximizar rendimiento (competiciones)

---

# PARTE 5: MÉTRICAS Y EVALUACIÓN DE MODELOS

## 5.1 Métricas para Clasificación

### Accuracy (Exactitud)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Predicciones correctas / Total predicciones
```

**Ejemplo: Detector de spam**
```
De 100 emails:
- 90 correctamente clasificados
- 10 incorrectamente clasificados

Accuracy = 90/100 = 90%
```

**Limitación**: No funciona bien con clases desbalanceadas

**Ejemplo del problema:**
```
Dataset: 95% clase A, 5% clase B
Modelo que siempre predice A: Accuracy = 95% (pero inútil para clase B!)
```

### Matriz de Confusión

La tabla fundamental para entender errores:

```
                    Predicho
                 Positivo  Negativo
Real  Positivo      TP        FN
      Negativo      FP        TN

TP = True Positives (Verdaderos Positivos)
TN = True Negatives (Verdaderos Negativos)
FP = False Positives (Falsos Positivos) - Error Tipo I
FN = False Negatives (Falsos Negativos) - Error Tipo II
```

**Ejemplo: Test médico para enfermedad**
```
                  Predicho
               Enfermo  Sano
Real Enfermo    90(TP)  10(FN)    ← FN: Falso negativo (¡peligroso!)
     Sano        20(FP) 880(TN)   ← FP: Falso positivo (alarma falsa)
```

### Precision (Precisión)
```
Precision = TP / (TP + FP)
```
"De todo lo que predijimos como positivo, ¿cuánto era realmente positivo?"

**Ejemplo:**
```
Detector de spam predice 100 emails como spam
De esos 100, 85 son realmente spam

Precision = 85/100 = 85%
```

**Cuándo es crítico**: Cuando FP son costosos (ej: marcar email importante como spam)

### Recall / Sensitivity (Sensibilidad / Exhaustividad)
```
Recall = TP / (TP + FN)
```
"De todos los positivos reales, ¿cuántos detectamos?"

**Ejemplo:**
```
Hay 90 emails de spam reales
El detector encontró 85 de ellos

Recall = 85/90 = 94.4%
```

**Cuándo es crítico**: Cuando FN son costosos (ej: no detectar fraude, no detectar cáncer)

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Media armónica de Precision y Recall. Balance entre ambos.

**Ejemplo:**
```
Precision = 85%
Recall = 94.4%

F1 = 2 × (0.85 × 0.944) / (0.85 + 0.944) = 89.3%
```

### Specificity (Especificidad)
```
Specificity = TN / (TN + FP)
```
"De todos los negativos reales, ¿cuántos identificamos correctamente?"

### Trade-offs importantes:

**Precision vs Recall:**
```
Threshold alto → Alta Precision, Bajo Recall
  (Solo predecimos positivo cuando muy seguros)

Threshold bajo → Baja Precision, Alto Recall
  (Predecimos positivo con cualquier sospecha)
```

**Ejemplo práctico - Filtro de spam:**
```
Configuración "Agresiva":
  Alto Recall (detecta 99% de spam)
  Baja Precision (muchos emails legítimos marcados como spam)

Configuración "Conservadora":
  Alta Precision (lo que marca es spam casi seguro)
  Bajo Recall (deja pasar algo de spam)
```

## 5.2 Métricas para Regresión

Cuando predecimos valores continuos (precio, temperatura, etc.):

### MSE (Mean Squared Error)
```
MSE = (1/n) Σ(y_real - y_pred)²
```
- Penaliza fuertemente errores grandes
- En unidades al cuadrado (difícil interpretar)
- Sensible a outliers

### RMSE (Root Mean Squared Error)
```
RMSE = √MSE = √[(1/n) Σ(y_real - y_pred)²]
```
- Mismas unidades que la variable objetivo
- Más interpretable que MSE
- Ejemplo: RMSE = $10,000 en predicción de precios de casas

### MAE (Mean Absolute Error)
```
MAE = (1/n) Σ|y_real - y_pred|
```
- Promedio de errores absolutos
- Menos sensible a outliers que MSE
- Fácil de interpretar

### R² (R-squared / Coeficiente de Determinación)
```
R² = 1 - (SS_residual / SS_total)
donde:
SS_residual = Σ(y_real - y_pred)²
SS_total = Σ(y_real - y_mean)²
```

**Interpretación:**
- R² = 1: Predicción perfecta
- R² = 0: Modelo tan bueno como predecir la media
- R² < 0: Peor que predecir la media

**Ejemplo:**
```
R² = 0.85 → El modelo explica el 85% de la varianza en los datos
```

### MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) Σ|((y_real - y_pred)/y_real)|
```
- Error en porcentaje
- Útil para comparar modelos en diferentes escalas
- Problema: indefinido cuando y_real = 0

## 5.3 Validación del Modelo

### División de Datos (Train/Validation/Test Split)

```
Dataset completo: 10,000 muestras

├── Entrenamiento (Train): 7,000 (70%)
│   └── Para entrenar el modelo
│
├── Validación (Validation): 1,500 (15%)
│   └── Para ajustar hiperparámetros y prevenir overfitting
│
└── Test: 1,500 (15%)
    └── Para evaluación final (¡NUNCA usar para entrenar!)
```

**Regla de oro**: El conjunto de test NUNCA debe verse durante desarrollo

### K-Fold Cross-Validation

```
Dataset dividido en K=5 partes (folds):

Iteración 1: [Test][Train][Train][Train][Train]
Iteración 2: [Train][Test][Train][Train][Train]
Iteración 3: [Train][Train][Test][Train][Train]
Iteración 4: [Train][Train][Train][Test][Train]
Iteración 5: [Train][Train][Train][Train][Test]

Métrica final: Promedio de las 5 iteraciones
```

**Ventajas:**
- Usa todos los datos para entrenar y validar
- Estimación más robusta del rendimiento
- Reduce varianza de la evaluación

**Cuándo usar:**
- Datasets pequeños (< 10,000 muestras)
- Cuando quieres máxima robustez
- No en deep learning (muy costoso computacionalmente)

### Overfitting vs Underfitting

```
Error de entrenamiento vs Error de validación:

UNDERFITTING:
Train error: ALTO
Val error: ALTO
→ Modelo muy simple, no captura patrones

BALANCE ÓPTIMO (¡meta!):
Train error: BAJO
Val error: BAJO
→ Generaliza bien

OVERFITTING:
Train error: MUY BAJO
Val error: ALTO
→ Memoriza datos de entrenamiento, no generaliza
```

**Señales de Overfitting:**
- Gran diferencia entre error de train y validation
- Accuracy 100% en train, 60% en validation
- Modelo funciona perfecto en desarrollo, mal en producción

**Señales de Underfitting:**
- Error alto en train y validation
- Modelo demasiado simple para la complejidad del problema

**Cómo combatir Overfitting:**
1. Más datos de entrenamiento
2. Regularización (L1, L2, Dropout)
3. Reducir complejidad del modelo
4. Early stopping
5. Data augmentation
6. Cross-validation

**Cómo combatir Underfitting:**
1. Modelo más complejo
2. Más features relevantes
3. Reducir regularización
4. Más tiempo de entrenamiento

---

# PARTE 6: CONCEPTOS AVANZADOS

## 7.2 Teoría de la Información

### Información Mutua
¿Cuánta información sobre B nos da conocer A?
```
I(X;Y) = H(X) - H(X|Y)
```

Si I(Jugada_actual; Jugada_anterior) > 0 → ¡Hay dependencia!

### Compresión y Predicción
"La mejor compresión es la mejor predicción"
- Si podemos comprimir el historial → hay patrones
- Historial aleatorio → incompresible

## 7.3 Aprendizaje Online vs Batch

### Batch Learning
- Entrena con todos los datos de una vez
- No se adapta durante el juego

### Online Learning
- Actualiza con cada nueva jugada
- Se adapta en tiempo real
- Nuestro proyecto usa esto

### Tasa de Aprendizaje
```
nuevo_peso = peso_anterior + α × (realidad - predicción)
```
- α alta: Aprende rápido, olvida rápido
- α baja: Aprende lento, memoria larga

---

# PARTE 8: ARQUITECTURA DE UN SISTEMA DE IA

## 8.1 Pipeline de Machine Learning

```
1. RECOLECCIÓN DE DATOS
   ↓
2. PREPROCESAMIENTO
   - Limpieza
   - Normalización
   - Feature Engineering
   ↓
3. ENTRENAMIENTO
   - Selección de algoritmo
   - Ajuste de hiperparámetros
   ↓
4. VALIDACIÓN
   - Métricas
   - Cross-validation
   ↓
5. PREDICCIÓN
   ↓
6. EVALUACIÓN Y MEJORA
   ↓
(Ciclo continuo)
```

## 8.2 Feature Engineering

**Features básicas**:
- Última jugada
- Frecuencias históricas
- Racha actual

**Features avanzadas**:
- Entropía últimas 10 jugadas
- Tiempo desde último cambio
- Patrón dominante actual
- Estado emocional inferido

## 8.3 Ensemble Methods

Combinar múltiples modelos para mejor resultado:

**Voting**:
```
Modelo1: Piedra
Modelo2: Piedra  → Mayoría: Piedra
Modelo3: Papel
```

**Stacking**:
```
Nivel 1: Modelos base hacen predicciones
Nivel 2: Meta-modelo decide basándose en predicciones nivel 1
```

**Boosting**:
Modelos secuenciales, cada uno corrige errores del anterior.

---

# PARTE 9: IMPLEMENTACIÓN PRÁCTICA

## 9.1 Estructura de Datos

### Representación del Estado
```python
estado = {
    'historial': ['piedra', 'papel', 'tijera', ...],
    'marcador': {'victorias': 10, 'derrotas': 8, 'empates': 5},
    'patrones_detectados': {('p','p'): 'tijera', ...},
    'tiempo_respuesta': [2.1, 1.8, 3.2, ...],
    'metadata': {'timestamp': ..., 'sesion': 1}
}
```

### Codificación
```
Piedra = 0, Papel = 1, Tijera = 2
One-hot: Piedra = [1,0,0], Papel = [0,1,0], Tijera = [0,0,1]
```

## 9.2 Algoritmo Principal

```
INICIALIZAR modelos
MIENTRAS juego_activo:
    estado = OBTENER_ESTADO_ACTUAL()
    
    PARA CADA modelo EN modelos:
        predicciones[modelo] = modelo.predecir(estado)
    
    decision = COMBINAR_PREDICCIONES(predicciones)
    resultado = JUGAR(decision)
    
    ACTUALIZAR_MODELOS(resultado)
    ACTUALIZAR_PESOS(rendimiento)
```

## 9.3 Optimizaciones

### Caché de Patrones
Guardar patrones ya calculados para no recalcular.

### Ventana Adaptativa
Ajustar tamaño de ventana según entropía actual.

### Decay de Memoria
Dar menos peso a jugadas muy antiguas:
```
peso = e^(-λ × antigüedad)
```

---

# PARTE 10: EVALUACIÓN EXPERIMENTAL

## 10.1 Diseño de Experimentos

### Variables Independientes
- Tipo de algoritmo
- Tamaño de ventana
- Longitud de patrones

### Variables Dependientes  
- Tasa de victoria
- Tiempo de convergencia
- Robustez ante cambios

### Variables de Control
- Número de jugadas
- Tipo de oponente
- Semilla aleatoria

## 10.2 Hipótesis a Probar

1. **H1**: Los patrones de longitud 3 predicen mejor que longitud 2
2. **H2**: La ventana adaptativa supera a la ventana fija
3. **H3**: El ensemble supera a cualquier modelo individual
4. **H4**: La precisión mejora logarítmicamente con los datos

## 10.3 Análisis Estadístico

### Test de Significancia
¿La diferencia entre modelos es estadísticamente significativa?
- t-test para comparar medias
- p-value < 0.05 → Significativo

### Intervalos de Confianza
"Con 95% confianza, la tasa de victoria está entre 42% y 48%"

### Análisis de Varianza (ANOVA)
Comparar múltiples modelos simultáneamente.

---

# EJERCICIOS TEÓRICOS

## Ejercicio 1: Probabilidad
Un jugador tiene estas estadísticas:
- Juega piedra 45% del tiempo
- Juega papel 35% del tiempo  
- Juega tijera 20% del tiempo

Calcula:
a) La probabilidad de que ganes si siempre juegas papel
b) Tu mejor estrategia fija
c) El valor esperado de tu mejor estrategia

## Ejercicio 2: Entropía
Calcula la entropía de estos jugadores:
a) [piedra, piedra, piedra, piedra, piedra]
b) [piedra, papel, tijera, piedra, papel]
c) ¿Cuál es más predecible?

## Ejercicio 3: Patrones
Dado el historial: [P, P, T, P, P, T, P, P, ?]
a) ¿Qué predices para '?'?
b) ¿Qué patrón identificas?
c) ¿Qué jugarías tú?

## Ejercicio 4: Markov
Construye la matriz de transición para:
[piedra, papel, papel, tijera, piedra, papel, tijera, tijera, piedra]

## Ejercicio 5: Teoría de Juegos
Demuestra por qué en PPT perfecto, la estrategia óptima es 1/3 para cada opción.

---

# CONEXIÓN CON EL PROYECTO

Ahora que entendéis la teoría, en vuestro proyecto vais a:

1. **Implementar** estos algoritmos en código Python
2. **Experimentar** con diferentes estrategias
3. **Medir** el rendimiento con las métricas aprendidas
4. **Optimizar** basándoos en los resultados
5. **Competir** para ver quién tiene el mejor modelo

La teoría os da las herramientas, ¡la práctica os dará la experiencia!

---

# RECURSOS ADICIONALES

## Libros Recomendados
- "Pattern Recognition and Machine Learning" - Bishop (avanzado)
- "The Elements of Statistical Learning" - Hastie et al. (intermedio)
- "Machine Learning for Absolute Beginners" - Theobald (básico)

## Papers Relevantes
- Shannon, C. (1948). "A Mathematical Theory of Communication"
- Nash, J. (1951). "Non-Cooperative Games"
- Sutton & Barto. "Reinforcement Learning: An Introduction"

## Cursos Online Gratuitos
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- Google's Machine Learning Crash Course

## Herramientas y Librerías
- **scikit-learn**: ML clásico en Python
- **pandas**: Análisis de datos
- **numpy**: Computación numérica
- **matplotlib**: Visualización

---

# CONCLUSIÓN

La IA y el Machine Learning no son magia, son **matemáticas aplicadas** + **datos** + **algoritmos inteligentes**.

En este proyecto de Piedra, Papel o Tijera, vais a experimentar con todos estos conceptos:
- **Probabilidad** para entender las jugadas
- **Estadística** para analizar patrones
- **Algoritmos** para hacer predicciones
- **Evaluación** para medir el éxito
- **Iteración** para mejorar continuamente

Lo más importante: La IA aprende de los datos, pero **vosotros** aprenderéis del proceso.

¡Que gane el mejor algoritmo! 🎮🤖
