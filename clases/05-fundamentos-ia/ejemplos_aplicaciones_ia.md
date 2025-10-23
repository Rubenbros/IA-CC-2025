# Ejemplos y Aplicaciones Prácticas de IA y Machine Learning

## Tabla de Contenidos
1. [Procesamiento de Lenguaje Natural (NLP)](#1-procesamiento-de-lenguaje-natural)
2. [Visión por Computadora](#2-visión-por-computadora)
3. [Sistemas de Recomendación](#3-sistemas-de-recomendación)
4. [Detección de Fraude](#4-detección-de-fraude)
5. [Predicción de Series Temporales](#5-predicción-de-series-temporales)
6. [Diagnóstico Médico Asistido](#6-diagnóstico-médico-asistido)

---

## 1. Procesamiento de Lenguaje Natural

### 1.1 Análisis de Sentimientos

**Problema**: Determinar si un texto expresa sentimiento positivo, negativo o neutral

**Ejemplo: Reviews de películas**

```
Datos de entrenamiento:
- "Esta película es increíble, la mejor del año" → POSITIVO
- "Aburrida y predecible, pérdida de tiempo" → NEGATIVO
- "La vi ayer en el cine" → NEUTRAL

Texto a clasificar:
- "Me encantó la actuación y la trama" → ¿?

Modelo predice: POSITIVO (95% confianza)
```

**Pipeline típico:**

1. **Preprocesamiento**:
   ```
   Texto original: "¡¡Esta película es INCREÍBLE!!"

   → Minúsculas: "¡¡esta película es increíble!!"
   → Remover puntuación: "esta película es increíble"
   → Tokenización: ["esta", "película", "es", "increíble"]
   → Remover stopwords: ["película", "increíble"]
   → Stemming/Lemmatization: ["pelicul", "incr"]
   ```

2. **Feature Extraction**:
   - **Bag of Words**: Contar frecuencia de palabras
   - **TF-IDF**: Importancia ponderada de palabras
   - **Word Embeddings**: Representación vectorial densa (Word2Vec, GloVe)

3. **Modelado**:
   - Naive Bayes (baseline)
   - Logistic Regression
   - LSTM / Transformers (modelos avanzados)

**Aplicaciones reales:**
- Análisis de feedback de clientes
- Monitoreo de redes sociales (crisis management)
- Sistemas de atención al cliente automáticos
- Análisis de mercado financiero (sentiment de noticias)

### 1.2 Chatbots y Asistentes Virtuales

**Componentes clave:**

1. **Intent Classification** (Clasificación de intención):
   ```
   Usuario: "¿Cuánto cuesta el envío a Madrid?"

   Intents posibles:
   - consulta_precio_envio (90% confianza) ✓
   - consulta_tiempo_entrega (5%)
   - consulta_producto (3%)
   ```

2. **Entity Recognition** (Reconocimiento de entidades):
   ```
   Texto: "Quiero reservar un vuelo a Barcelona el 15 de mayo"

   Entidades detectadas:
   - Acción: "reservar"
   - Tipo: "vuelo"
   - Destino: "Barcelona" [UBICACIÓN]
   - Fecha: "15 de mayo" [FECHA]
   ```

3. **Response Generation** (Generación de respuesta):
   - Rule-based: Plantillas predefinidas
   - Retrieval-based: Seleccionar de respuestas existentes
   - Generative: GPT-style (generar respuesta nueva)

**Ejemplo completo:**

```
Usuario: "Hola, quiero devolver un producto"

1. Intent: solicitud_devolucion (92%)
2. Entities: producto (genérico)
3. Dialog State: inicio_devolucion
4. Respuesta: "Claro, ¿puedes indicarme el número de pedido?"

Usuario: "Es el pedido #12345"

1. Intent: proporciona_info (88%)
2. Entities: numero_pedido = "12345"
3. Action: buscar_pedido(12345)
4. Respuesta: "Encontré tu pedido. Compraste [producto].
              ¿Cuál es el motivo de la devolución?"
```

---

## 2. Visión por Computadora

### 2.1 Clasificación de Imágenes

**Problema**: Identificar qué hay en una imagen

**Dataset clásico: MNIST (dígitos escritos a mano)**

```
Entrada: Imagen 28x28 píxeles en escala de grises
Salida: Dígito del 0 al 9

Ejemplo:
Input: [array de 784 valores de píxeles]
       [[0, 0, 0, 45, 120, ...],
        [0, 0, 80, 200, 255, ...],
        ...]

Output: "7" (con 98% de confianza)
```

**Arquitectura típica: Red Neuronal Convolucional (CNN)**

```
Input Image (28x28x1)
    ↓
Conv Layer 1 (32 filtros, 3x3)
    ↓
ReLU Activation
    ↓
Max Pooling (2x2)
    ↓
Conv Layer 2 (64 filtros, 3x3)
    ↓
ReLU Activation
    ↓
Max Pooling (2x2)
    ↓
Flatten
    ↓
Dense Layer (128 neuronas)
    ↓
Dropout (0.5)
    ↓
Dense Layer (10 neuronas - salida)
    ↓
Softmax → Probabilidades para cada dígito
```

**Métricas de éxito:**
- Accuracy en MNIST: >99% (estado del arte)
- Accuracy en ImageNet (1000 clases): ~80% (modelos modernos)

### 2.2 Detección de Objetos

**Problema**: No solo "qué" hay en la imagen, sino "dónde"

**Ejemplo: Detección de peatones en conducción autónoma**

```
Input: Imagen de la calle
Output:
  - Bounding boxes: coordenadas [x, y, ancho, alto]
  - Clases: ["peatón", "coche", "bicicleta", "señal"]
  - Confianzas: [0.95, 0.88, 0.76, 0.92]

Visualización:
[Imagen con cajas delimitadoras]
┌─────────┐
│ Peatón  │ 95%
└─────────┘
    ┌──────────┐
    │  Coche   │ 88%
    └──────────┘
```

**Algoritmos populares:**
- **YOLO (You Only Look Once)**: Muy rápido, tiempo real
- **Faster R-CNN**: Más preciso, más lento
- **SSD (Single Shot Detector)**: Balance velocidad/precisión

**Aplicaciones:**
- Conducción autónoma
- Vigilancia y seguridad
- Conteo de personas (retail analytics)
- Inspección industrial (detección de defectos)

### 2.3 Segmentación Semántica

**Problema**: Clasificar cada píxel de la imagen

**Ejemplo: Segmentación de imágenes médicas**

```
Input: Resonancia magnética del cerebro

Output: Máscara de segmentación
  - Píxeles rojos: Tumor
  - Píxeles azules: Tejido sano
  - Píxeles verdes: Fluido cerebroespinal

Cada píxel tiene una etiqueta de clase
```

**Arquitectura: U-Net (popular en imágenes médicas)**

```
Encoder (contracción):     Decoder (expansión):
Input                           Output
  ↓                              ↑
Conv + Pool    ----skip--->   Upsampling + Conv
  ↓                              ↑
Conv + Pool    ----skip--->   Upsampling + Conv
  ↓                              ↑
Bottleneck
```

---

## 3. Sistemas de Recomendación

### 3.1 Filtrado Colaborativo

**Principio**: "A los usuarios similares a ti les gustó X"

**Ejemplo: Recomendación de películas**

**Datos de entrenamiento: Matriz Usuario-Película**

```
            Avengers  Titanic  Matrix  StarWars
Usuario1       5        2       4        5
Usuario2       4        1       5        4
Usuario3       1        5       2        1
Usuario4       ?        ?       ?        5
```

**Algoritmo User-Based:**

1. Encontrar usuarios similares a Usuario4
   ```
   Similitud con Usuario1: Alta (ambos valoran StarWars alto)
   Similitud con Usuario2: Alta
   Similitud con Usuario3: Baja
   ```

2. Predecir ratings basándose en usuarios similares
   ```
   Usuario4 probablemente valorará:
   - Avengers: ~4.5 (alta confianza)
   - Matrix: ~4.5 (alta confianza)
   - Titanic: ~1.5 (baja probabilidad de gustar)
   ```

**Algoritmo Item-Based:**

```
Usuario4 le gusta StarWars

Películas similares a StarWars:
1. Avengers (similitud: 0.85)
2. Matrix (similitud: 0.78)
3. Titanic (similitud: 0.12)

Recomendación: Avengers, Matrix
```

### 3.2 Filtrado Basado en Contenido

**Principio**: "Te gusta X, entonces te gustará Y que es similar"

**Ejemplo: Recomendación de música**

```
Usuario escucha:
- Canción A: [género: rock, tempo: 120bpm, energía: alta]
- Canción B: [género: rock, tempo: 115bpm, energía: alta]

Perfil del usuario:
- Prefiere: rock, tempo medio-alto, energía alta

Candidatos a recomendar:
- Canción C: [rock, 125bpm, alta] → Similitud: 95% ✓
- Canción D: [jazz, 90bpm, baja] → Similitud: 30% ✗
```

**Feature Extraction en contenido:**

Para películas:
- Género, director, actores, año
- Palabras clave del plot
- Duración, presupuesto, ratings

Para música:
- Features de audio (tempo, tono, ritmo)
- Género, artista, época
- Análisis espectral

### 3.3 Sistemas Híbridos

**Ejemplo: Netflix**

Combina múltiples enfoques:

1. **Collaborative Filtering**:
   "Usuarios como tú vieron Y después de ver X"

2. **Content-Based**:
   "Te gustan las comedias románticas de los 90"

3. **Popularity-Based**:
   "Trending ahora en tu país"

4. **Deep Learning**:
   Redes neuronales que aprenden representaciones complejas

5. **Context-Aware**:
   - Hora del día (noche → películas cortas)
   - Dispositivo (móvil vs TV)
   - Día de la semana (fin de semana → películas largas)

---

## 4. Detección de Fraude

### 4.1 Fraude en Transacciones Bancarias

**Problema**: Identificar transacciones fraudulentas en tiempo real

**Características del problema:**
- **Desbalanceo extremo**: 99.9% transacciones legítimas, 0.1% fraude
- **Tiempo real**: Decisión en milisegundos
- **Alto costo de errores**:
  - Falso positivo (FP): Cliente molesto
  - Falso negativo (FN): Pérdida económica

**Features importantes:**

```python
features = {
    # Transacción actual
    'monto': 1500.00,
    'comercio_tipo': 'electrónica',
    'ubicacion': 'Madrid',
    'hora': '03:00 AM',  # ¡Sospechoso!

    # Histórico del usuario
    'monto_promedio_30dias': 150.00,  # ¡Outlier!
    'num_transacciones_hoy': 15,  # ¡Anormal!
    'ubicacion_habitual': 'Barcelona',  # ¡Diferente!
    'ultima_compra_minutos': 5,  # ¡Muy rápido!

    # Comportamiento del comercio
    'tasa_fraude_comercio': 0.15,  # ¡Alto riesgo!
    'país_comercio': 'Desconocido',
}
```

**Sistema de scoring:**

```
Modelo calcula risk score: 0-100

Score 0-30: VERDE → Aprobar automáticamente
Score 30-70: AMARILLO → Verificación adicional (SMS/OTP)
Score 70-100: ROJO → Bloquear y notificar

Ejemplo:
Transacción actual: Score = 85
  - Monto 10x superior a promedio: +30 puntos
  - Hora inusual (3 AM): +20 puntos
  - Ubicación diferente a habitual: +25 puntos
  - Comercio de alto riesgo: +10 puntos

Decisión: BLOQUEAR y llamar al cliente
```

**Técnicas avanzadas:**

1. **Anomaly Detection** (Isolation Forest, Autoencoders)
   ```
   Aprender qué es "normal" para cada usuario
   Detectar desviaciones significativas
   ```

2. **Graph Analytics**:
   ```
   Analizar redes de transacciones
   Detectar anillos de fraude coordinados

   Usuario A → Comercio X ← Usuario B → Comercio Y ← Usuario A
                          ↑________↓
                     ¡Patrón circular sospechoso!
   ```

3. **Behavioral Biometrics**:
   ```
   Analizar CÓMO el usuario interactúa:
   - Velocidad de tipeo
   - Patrones de movimiento del ratón
   - Presión en pantalla táctil
   ```

### 4.2 Detección de Bots y Spam

**Ejemplo: Filtro de spam de email**

**Features textuales:**
```python
email_features = {
    # Contenido
    'palabras_spam': 15,  # "gratis", "oferta", "urgente"...
    'exclamaciones': 8,  # "!!!!"
    'mayusculas_ratio': 0.6,  # "COMPRA AHORA"
    'enlaces': 5,
    'enlaces_acortados': 3,  # bit.ly, etc.

    # Metadatos
    'remitente_conocido': False,
    'dominio_reputacion': 0.2,  # Malo
    'hora_envio': '04:30 AM',  # Sospechoso

    # Estilísticos
    'errores_ortograficos': 12,
    'urgencia_palabras': 7,  # "ahora", "última oportunidad"
    'personalizado': False,  # No usa tu nombre
}
```

**Pipeline de detección:**

```
1. Filtros de reglas (rápido):
   - IP en blacklist → SPAM
   - Dominio verificado conocido → NO SPAM

2. Machine Learning (para casos ambiguos):
   - Naive Bayes: P(spam | contenido)
   - SVM con TF-IDF features
   - Deep Learning: BERT para entender contexto

3. Feedback loop:
   - Usuario marca como spam → Reentrenar modelo
   - Adaptación continua a nuevas técnicas de spam
```

---

## 5. Predicción de Series Temporales

### 5.1 Predicción de Demanda

**Problema**: Predecir ventas futuras para optimizar inventario

**Ejemplo: Ventas de retail**

**Datos históricos:**
```
Fecha        Ventas  Temp  Día_Semana  Promoción  Festivo
2024-01-01   1200    15°C  Lunes       No         Sí
2024-01-02   2500    16°C  Martes      Sí         No
2024-01-03   1800    14°C  Miércoles   No         No
...
```

**Features para el modelo:**

1. **Temporales**:
   - Día de la semana (efecto weekend)
   - Mes del año (estacionalidad)
   - Día del mes (cobro de nóminas)
   - Festivos

2. **Lagged variables** (valores pasados):
   ```
   ventas_ayer
   ventas_hace_7dias  # Mismo día semana anterior
   ventas_hace_30dias  # Mismo día mes anterior
   ```

3. **Rolling statistics**:
   ```
   media_movil_7dias = promedio(ventas últimos 7 días)
   media_movil_30dias
   desviacion_estandar_7dias  # Volatilidad
   ```

4. **Externas**:
   - Clima
   - Promociones
   - Eventos especiales
   - Competencia

**Modelos típicos:**

**1. ARIMA (AutoRegressive Integrated Moving Average)**
```
Componentes:
- AR: Ventas_hoy dependen de ventas_ayer
- I: Diferenciación para hacer serie estacionaria
- MA: Promedio de errores pasados

Ejemplo:
ARIMA(1,1,1):
  ventas_t = ventas_{t-1} + error_t + θ·error_{t-1}
```

**2. Prophet (Facebook)**
```
Descompone serie en:
- Tendencia (crecimiento general)
- Estacionalidad anual
- Estacionalidad semanal
- Festivos
- Eventos especiales

y = tendencia + estacionalidad + festivos + error
```

**3. LSTM (Deep Learning)**
```
Secuencias de entrada → LSTM → Predicción

Input: [ventas_día1, ventas_día2, ..., ventas_día30]
Output: ventas_día31

Aprende patrones temporales complejos
```

**Métricas de evaluación:**

```
Predicción vs Real:
MAE (Mean Absolute Error):
  promedio(|predicho - real|) = 120 unidades

MAPE (Mean Absolute Percentage Error):
  promedio(|predicho - real| / real) = 8.5%

Interpretación:
"En promedio, nos equivocamos en 120 unidades (8.5%)"
```

### 5.2 Predicción de Precios de Acciones

**Problema**: Predecir movimiento del precio de acciones

⚠️ **Advertencia**: Mercados son altamente impredecibles. No hay "bala de plata".

**Features técnicas:**

```python
features = {
    # Precio
    'precio_actual': 150.0,
    'precio_apertura': 148.5,
    'precio_max_dia': 152.0,
    'precio_min_dia': 147.0,

    # Volumen
    'volumen': 1_500_000,
    'volumen_promedio_20dias': 1_200_000,

    # Indicadores técnicos
    'SMA_20': 149.0,  # Simple Moving Average
    'SMA_50': 145.0,
    'RSI': 65,  # Relative Strength Index (0-100)
    'MACD': 2.5,  # Moving Average Convergence Divergence
    'Bollinger_upper': 155.0,
    'Bollinger_lower': 143.0,

    # Sentimiento
    'sentiment_news': 0.6,  # Análisis de noticias
    'sentiment_twitter': 0.4,

    # Macroeconómicos
    'S&P500_cambio': 0.02,
    'tasa_interes': 0.05,
    'VIX': 18,  # Volatilidad del mercado
}
```

**Objetivo**: ¿Subirá o bajará en próximas X horas/días?

```
Clasificación binaria:
- Clase 0: Bajará
- Clase 1: Subirá

O Regresión:
- Predecir precio exacto en T+1
```

**Limitaciones importantes:**

1. **Hipótesis del Mercado Eficiente**: Toda información pública ya está en el precio
2. **Ruido vs Señal**: Mucha aleatoriedad, poca señal predecible
3. **No-estacionariedad**: Distribución cambia constantemente
4. **Cisnes negros**: Eventos impredecibles (COVID, guerras, etc.)

**Enfoque realista:**

```
No intentar predecir el futuro exactamente, sino:
1. Identificar patrones de mayor probabilidad
2. Gestión de riesgo (stop-loss, diversificación)
3. Combinar con análisis fundamental
4. Backtesting riguroso en datos históricos
```

---

## 6. Diagnóstico Médico Asistido

### 6.1 Detección de Cáncer en Imágenes

**Ejemplo: Detección de cáncer de piel (melanoma)**

**Input**: Imagen dermatoscópica de lunar

**Output**:
- Probabilidad de ser maligno vs benigno
- Tipo específico si es maligno

**Dataset de entrenamiento:**
```
10,000 imágenes etiquetadas por dermatólogos:
- 7,000 benignos
- 3,000 malignos (varios tipos)

Características de cada imagen:
- Asimetría
- Bordes irregulares
- Color no uniforme
- Diámetro
- Evolución (comparar con imágenes previas)
```

**Arquitectura: Transfer Learning con CNN**

```
Modelo pre-entrenado (ImageNet):
  ResNet-50 / InceptionV3
    ↓
Congelar capas iniciales (features generales)
    ↓
Fine-tuning en capas finales con datos médicos
    ↓
Capas personalizadas:
  Dense(512) → Dropout(0.5) → Dense(2, softmax)
    ↓
Output: [P(benigno), P(maligno)]
```

**Métricas cruciales:**

```
Matriz de Confusión:
                 Predicho
              Benigno  Maligno
Real Benigno    920      80     ← FP: Alarma falsa
     Maligno     15     285     ← FN: ¡MUY PELIGROSO!

Métricas:
- Sensitivity (Recall): 285/(285+15) = 95%
  "De 100 casos malignos, detectamos 95"

- Specificity: 920/(920+80) = 92%
  "De 100 casos benignos, identificamos correctamente 92"

Para diagnóstico médico:
  PRIORIDAD: Minimizar FN (no queremos perder casos malignos)
  Estrategia: Threshold bajo → Alta sensitivity (>95%)
```

**Implementación en clínica:**

```
Sistema de asistencia (NO reemplazo):

1. IA hace screening inicial
   ↓
2. Casos de alto riesgo → Prioridad en agenda del dermatólogo
   ↓
3. Médico hace diagnóstico final
   ↓
4. Feedback al sistema (mejora continua)

Resultado:
- Médicos pueden ver más pacientes
- Detección temprana aumenta
- Reducción de falsos negativos
```

### 6.2 Predicción de Readmisión Hospitalaria

**Problema**: ¿Qué pacientes tienen alto riesgo de reingresar en 30 días?

**Objetivo**: Intervención preventiva para evitar readmisión

**Features del paciente:**

```python
paciente = {
    # Demográficos
    'edad': 72,
    'genero': 'M',

    # Historial
    'num_hospitalizaciones_previas': 3,
    'diagnostico_principal': 'insuficiencia_cardiaca',
    'comorbilidades': ['diabetes', 'hipertension'],
    'num_medicamentos': 12,

    # Estancia actual
    'dias_hospitalizacion': 8,
    'procedimientos_realizados': 3,
    'complicaciones': True,

    # Al alta
    'adherencia_medicacion_estimada': 0.6,  # Baja
    'soporte_familiar': 'bajo',
    'tiene_cita_seguimiento': False,  # ¡Riesgo!
    'distancia_hospital_km': 45,

    # Clínicos
    'presion_arterial': [160, 95],  # Alta
    'frecuencia_cardiaca': 95,
    'saturacion_oxigeno': 92,  # Baja
}
```

**Modelo predictivo:**

```
Gradient Boosting Classifier:

Input: Features del paciente
Output: P(readmision_30dias)

Ejemplo:
Paciente A: P(readmision) = 0.75 (75%)  ← ALTO RIESGO

Feature importance:
1. adherencia_medicacion: 25%
2. num_hospitalizaciones_previas: 18%
3. complicaciones: 15%
4. tiene_cita_seguimiento: 12%
...
```

**Intervención basada en riesgo:**

```
Riesgo < 30%: Alta estándar

Riesgo 30-60%:
  - Llamada telefónica a los 7 días
  - Educación sobre señales de alerta

Riesgo > 60%:
  - Visita de enfermería a domicilio (48h)
  - Cita médica en 1 semana (no 1 mes)
  - Programa de adherencia a medicación
  - Coordinación con farmacia local
```

**Impacto medido:**

```
Antes del sistema:
- Tasa de readmisión: 18%

Después del sistema:
- Tasa de readmisión: 12%
- Reducción: 33%

Beneficios:
- Menos sufrimiento del paciente
- Ahorro económico al hospital
- Mejor uso de recursos
```

---

## Conclusiones Generales

### Patrón común en todos los proyectos de ML:

1. **Definir el problema claramente**
   - ¿Qué queremos predecir?
   - ¿Cómo medimos el éxito?
   - ¿Qué datos necesitamos?

2. **Recolectar y explorar datos**
   - Análisis exploratorio (EDA)
   - Visualizaciones
   - Entender distribuciones y correlaciones

3. **Preparar datos**
   - Limpieza
   - Manejo de valores faltantes
   - Feature engineering
   - Normalización/Estandarización

4. **Entrenar modelos**
   - Empezar simple (baseline)
   - Probar múltiples algoritmos
   - Validación cruzada

5. **Evaluar y comparar**
   - Métricas apropiadas al problema
   - Análisis de errores
   - Importancia de features

6. **Optimizar**
   - Hyperparameter tuning
   - Ensemble methods
   - Feature selection

7. **Deploy y monitorear**
   - Poner en producción
   - Monitorear rendimiento
   - Reentrenar periódicamente

### Lecciones clave:

✅ **Más datos > Mejor algoritmo** (hasta cierto punto)

✅ **Feature engineering es crucial**: La representación de datos importa más que el algoritmo

✅ **No hay almuerzo gratis**: No existe un algoritmo que funcione mejor para todo

✅ **Interpretabilidad vs Accuracy**: A veces un modelo simple y explicable es mejor que una caja negra

✅ **Datos desbalanceados**: Requieren técnicas especiales (oversampling, undersampling, pesos)

✅ **Validación rigurosa**: Train/Validation/Test separados estrictamente

✅ **Ética y sesgo**: Modelos pueden perpetuar sesgos presentes en datos

✅ **Mantenimiento continuo**: Los modelos se degradan con el tiempo (data drift)

---

## Recursos para Practicar

### Datasets públicos:
- **Kaggle**: Miles de datasets y competiciones
- **UCI ML Repository**: Datasets clásicos
- **Google Dataset Search**: Buscador de datasets
- **Data.gov**: Datos gubernamentales

### Competiciones:
- **Kaggle Competitions**: Desde principiante hasta experto
- **DrivenData**: Enfoque social good
- **AIcrowd**: Challenges variados

### Proyectos sugeridos para empezar:

**Nivel Principiante:**
1. Predicción de precios de casas (regresión)
2. Clasificación de dígitos MNIST
3. Análisis de sentimientos en reviews
4. Predicción de supervivencia del Titanic

**Nivel Intermedio:**
5. Sistema de recomendación de películas
6. Detección de spam en emails
7. Predicción de abandono de clientes (churn)
8. Clasificación de imágenes de ropa (Fashion MNIST)

**Nivel Avanzado:**
9. Chatbot con NLP
10. Detección de objetos en video
11. Trading algorítmico
12. Generación de texto con transformers

¡El mejor aprendizaje viene de hacer proyectos reales!
