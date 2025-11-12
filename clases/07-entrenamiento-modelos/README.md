# Clase 07: Entrenamiento de Modelos de Machine Learning

## Índice
1. [Introducción](#introducción)
2. [El Proceso de Entrenamiento](#el-proceso-de-entrenamiento)
3. [Train/Test Split](#traintest-split)
4. [Tipos de Modelos](#tipos-de-modelos)
5. [Scikit-learn Básico](#scikit-learn-básico)
6. [Evaluación de Modelos](#evaluación-de-modelos)
7. [Overfitting y Underfitting](#overfitting-y-underfitting)
8. [Aplicación a Piedra, Papel o Tijera](#aplicación-a-piedra-papel-o-tijera)

---

## Introducción

Después de aprender a preparar nuestros datos con **Feature Engineering**, el siguiente paso es **entrenar modelos** que aprendan patrones y puedan hacer predicciones.

### ¿Qué es entrenar un modelo?

**Entrenar** es el proceso de mostrarle a un algoritmo muchos ejemplos para que aprenda patrones.

```
DATOS + ALGORITMO = MODELO ENTRENADO
```

**Analogía**: Es como enseñarle a un niño a reconocer animales:
- Le muestras 100 fotos de perros y gatos
- El niño aprende las diferencias (orejas, tamaño, forma)
- Después puede identificar nuevos animales que nunca vio

---

## El Proceso de Entrenamiento

### Flujo completo de Machine Learning

```
1. DATOS CRUDOS
   ↓
2. FEATURE ENGINEERING (Clase 06)
   - Limpiar datos
   - Crear features
   - Normalizar
   ↓
3. DIVISIÓN DE DATOS (Esta clase)
   - Train set (80%)
   - Test set (20%)
   ↓
4. ENTRENAMIENTO (Esta clase)
   - Elegir algoritmo
   - Entrenar con train set
   ↓
5. EVALUACIÓN (Esta clase)
   - Probar con test set
   - Calcular métricas
   ↓
6. PREDICCIÓN
   - Usar en datos nuevos
```

---

## Train/Test Split

### ¿Por qué dividir los datos?

**Problema**: Si entrenas y evalúas con los mismos datos, el modelo puede "memorizar" en lugar de "aprender".

**Solución**: Dividir en dos conjuntos:

1. **Train Set (80%)**: Datos para entrenar
2. **Test Set (20%)**: Datos para evaluar (el modelo nunca los ve durante entrenamiento)

### Analogía: Examen de matemáticas

```
Train Set = Ejercicios de práctica
Test Set  = Examen final (problemas nuevos)
```

Si practicas con los mismos problemas del examen, no sabes si realmente aprendiste matemáticas o solo memorizaste las respuestas.

### Código en Python

```python
from sklearn.model_selection import train_test_split

# Separar features (X) y target (y)
X = df[['feature1', 'feature2', 'feature3']]  # Features
y = df['resultado']                            # Lo que queremos predecir

# Dividir 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para test
    random_state=42     # Para reproducibilidad
)

print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de test: {len(X_test)}")
```

---

## Tipos de Modelos

En Machine Learning, los modelos se agrupan según el **tipo de predicción** que hacen. Los dos tipos principales son:

### 1. Regresión (predecir números)

**Definición**: Modelos que predicen un **valor numérico continuo**.

**Cuándo usar**: Cuando tu variable objetivo (lo que quieres predecir) puede tomar cualquier valor numérico en un rango.

**Características**:
- La salida es un número decimal o entero
- Puede haber infinitos valores posibles
- El modelo aprende una función matemática que mapea inputs → número

**Ejemplos del mundo real**:
- **Precio de una casa**: 150,000€, 250,500€, 320,750€
  - Features: metros cuadrados, habitaciones, ubicación
  - Target: precio (número continuo)
- **Temperatura de mañana**: 18.5°C, 22.3°C, 15.8°C
  - Features: temperatura hoy, presión, humedad
  - Target: temperatura (número continuo)
- **Ventas del próximo mes**: 1250 unidades, 1875 unidades
  - Features: ventas anteriores, promociones, temporada
  - Target: cantidad vendida (número)

**Analogía**: Es como pedir a alguien "¿Cuánto pesará el bebé al nacer?" → La respuesta es un número específico (3.2 kg, 3.5 kg, etc.)

#### Algoritmos comunes de Regresión:

**a) Regresión Lineal** (`LinearRegression`)
- **Idea**: Traza una línea recta que mejor se ajuste a los datos
- **Fórmula**: `y = mx + b` (como en matemáticas)
- **Cuándo usar**: Cuando la relación es aproximadamente lineal
- **Ejemplo**: Predecir precio según metros cuadrados (más metros → más caro)

```python
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(X_train, y_train)
```

**b) Regresión Polinomial** (`PolynomialFeatures + LinearRegression`)
- **Idea**: Curvas en lugar de líneas rectas
- **Fórmula**: `y = ax² + bx + c`
- **Cuándo usar**: Cuando la relación no es lineal (ej: crecimiento que acelera)

**c) Random Forest Regressor** (`RandomForestRegressor`)
- **Idea**: Muchos árboles de decisión votando un número
- **Cuándo usar**: Relaciones complejas, muchas features
- **Ventaja**: Captura patrones no lineales complejos

```python
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=100)
modelo.fit(X_train, y_train)
```

**d) Support Vector Regression (SVR)**
- **Idea**: Encuentra un "tubo" que contenga la mayoría de puntos
- **Cuándo usar**: Datos con ruido, relaciones complejas

---

### 2. Clasificación (predecir categorías)

**Definición**: Modelos que predicen una **categoría o etiqueta discreta**.

**Cuándo usar**: Cuando tu variable objetivo es una clase o grupo específico (NO un número continuo).

**Características**:
- La salida es una etiqueta o clase
- Número finito de opciones posibles
- El modelo aprende fronteras de decisión entre clases

**Tipos de clasificación**:
1. **Clasificación binaria**: Solo 2 clases
   - Spam vs No spam
   - Enfermo vs Sano
   - Aprobar vs Reprobar

2. **Clasificación multiclase**: 3+ clases
   - Dígitos 0-9 (10 clases)
   - Tipos de flores (3 clases)
   - **Piedra, Papel o Tijera** (3 clases) ← Nuestro proyecto

**Ejemplos del mundo real**:
- **Email spam o no spam**
  - Features: palabras clave, remitente, enlaces
  - Target: "spam" o "no spam" (2 clases)
- **Reconocer dígitos escritos a mano**
  - Features: píxeles de la imagen
  - Target: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 clases)
- **Diagnóstico médico**
  - Features: síntomas, edad, análisis
  - Target: "saludable", "gripe", "neumonía" (3 clases)
- **Predecir jugada en PPT**
  - Features: historial, frecuencias, rachas
  - Target: "piedra", "papel", "tijera" (3 clases)

**Analogía**: Es como preguntar "¿De qué color es esta fruta?" → La respuesta es una categoría (rojo, verde, amarillo), NO un número.

#### Algoritmos comunes de Clasificación:

**a) K-Nearest Neighbors (KNN)** (`KNeighborsClassifier`)
- **Idea**: "Dime con quién andas y te diré quién eres"
- **Cómo funciona**:
  1. Busca los K ejemplos más similares en el entrenamiento
  2. Mira qué clase es más común entre esos K vecinos
  3. Predice esa clase
- **Ejemplo**: Si los 5 vecinos más cercanos son [piedra, piedra, papel, piedra, tijera] → predice "piedra"
- **Cuándo usar**: Patrones locales, fronteras irregulares
- **Hiperparámetro clave**: `n_neighbors` (cuántos vecinos considerar)

```python
from sklearn.neighbors import KNeighborsClassifier
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)
```

**b) Decision Trees (Árboles de Decisión)** (`DecisionTreeClassifier`)
- **Idea**: Serie de preguntas Si-Entonces
- **Cómo funciona**:
  ```
  ¿freq_piedra > 0.4?
  ├─ SÍ: ¿ultima_jugada = piedra?
  │  ├─ SÍ: Predecir PAPEL
  │  └─ NO: Predecir PIEDRA
  └─ NO: Predecir TIJERA
  ```
- **Cuándo usar**: Quieres interpretabilidad, capturar reglas
- **Ventaja**: Muy fácil de visualizar y explicar
- **Desventaja**: Propenso a overfitting

```python
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(max_depth=5)
modelo.fit(X_train, y_train)
```

**c) Random Forest** (`RandomForestClassifier`)
- **Idea**: Bosque de muchos árboles votando
- **Cómo funciona**:
  1. Crea 100 árboles diferentes (cada uno ve datos aleatorios)
  2. Cada árbol vota por una clase
  3. La clase más votada gana
- **Ejemplo**: 60 árboles dicen "piedra", 30 dicen "papel", 10 dicen "tijera" → predice "piedra"
- **Cuándo usar**: Cuando quieres alta precisión sin preocuparte mucho por hiperparámetros
- **Ventaja**: Muy robusto, reduce overfitting, excelente rendimiento
- **Desventaja**: Más lento, menos interpretable

```python
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, max_depth=5)
modelo.fit(X_train, y_train)
```

**d) Logistic Regression** (`LogisticRegression`)
- **Nombre confuso**: A pesar del nombre, es CLASIFICACIÓN, no regresión
- **Idea**: Calcula probabilidad de cada clase y elige la más alta
- **Cómo funciona**: Usa función logística para mapear features → probabilidad [0, 1]
- **Ejemplo**: P(piedra)=0.6, P(papel)=0.3, P(tijera)=0.1 → predice "piedra"
- **Cuándo usar**: Clasificación binaria, cuando quieres probabilidades
- **Ventaja**: Rápido, simple, da probabilidades

```python
from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
```

**e) Support Vector Machines (SVM)** (`SVC`)
- **Idea**: Encuentra el "mejor hiperplano" que separa las clases
- **Cómo funciona**: Maximiza el margen entre clases
- **Cuándo usar**: Datasets pequeños-medianos, clasificación binaria
- **Ventaja**: Efectivo en espacios de alta dimensión
- **Desventaja**: Lento con muchos datos

```python
from sklearn.svm import SVC
modelo = SVC(kernel='rbf')
modelo.fit(X_train, y_train)
```

**f) Naive Bayes** (`GaussianNB`, `MultinomialNB`)
- **Idea**: Usa teorema de Bayes (probabilidades condicionales)
- **Cuándo usar**: Clasificación de texto (spam, sentimientos)
- **Ventaja**: Muy rápido, funciona bien con pocas muestras
- **Desventaja**: Asume que features son independientes (raramente cierto)

---

### ¿Cómo elegir entre Regresión y Clasificación?

**Pregunta clave**: ¿Tu variable objetivo es un número continuo o una categoría?

| Pregunta | Respuesta | Tipo |
|----------|-----------|------|
| ¿Cuánto costará? | 125,000€ | Regresión |
| ¿Qué tipo es? | "deportivo" | Clasificación |
| ¿Cuántos días? | 5.2 días | Regresión |
| ¿Aprueba o reprueba? | "aprueba" | Clasificación |
| ¿Qué temperatura? | 22.3°C | Regresión |
| ¿Qué jugará? | "piedra" | Clasificación ← PPT |

**Truco mental**:
- Si puedes decir "entre 23 y 24" → Regresión
- Si solo hay opciones específicas → Clasificación

---

### 3. Otros tipos (menciones honoríficas)

Aunque en esta clase nos enfocamos en Clasificación (para PPT), existen otros tipos:

**Clustering** (Agrupamiento)
- **Qué hace**: Agrupa datos similares sin etiquetas previas
- **Ejemplo**: Segmentar clientes por comportamiento
- **No supervisado**: No necesita etiquetas de entrenamiento
- **Algoritmo**: K-Means

**Reducción de dimensionalidad**
- **Qué hace**: Reduce número de features manteniendo información
- **Ejemplo**: Visualizar datos de 100 dimensiones en 2D
- **Algoritmo**: PCA (Principal Component Analysis)

**Detección de anomalías**
- **Qué hace**: Encuentra datos "raros" o fuera de lo común
- **Ejemplo**: Detectar fraude en tarjetas de crédito
- **Algoritmo**: Isolation Forest

---

## Scikit-learn Básico

### ¿Qué es scikit-learn?

La biblioteca más popular de Machine Learning en Python. Todos los modelos siguen la misma interfaz:

```python
# 1. IMPORTAR
from sklearn.neighbors import KNeighborsClassifier

# 2. CREAR
modelo = KNeighborsClassifier(n_neighbors=5)

# 3. ENTRENAR
modelo.fit(X_train, y_train)

# 4. PREDECIR
predicciones = modelo.predict(X_test)

# 5. EVALUAR
accuracy = modelo.score(X_test, y_test)
print(f"Precisión: {accuracy:.2%}")
```

### Ventaja: API consistente

**Todos** los modelos de scikit-learn usan estos mismos métodos:
- `.fit(X, y)` → entrenar
- `.predict(X)` → predecir
- `.score(X, y)` → evaluar

Esto hace fácil probar diferentes algoritmos cambiando solo 1 línea.

---

## Evaluación de Modelos

### Métricas para Clasificación

#### 1. Accuracy (Precisión)

**Definición**: Porcentaje de predicciones correctas.

```
Accuracy = (Predicciones correctas) / (Total de predicciones)
```

**Ejemplo**:
- Hicimos 100 predicciones
- 85 fueron correctas
- Accuracy = 85/100 = 0.85 = 85%

**Código**:
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicciones)
print(f"Accuracy: {accuracy:.2%}")
```

#### 2. Matriz de Confusión

Muestra dónde se equivoca el modelo:

```
                  PREDICCIÓN
                Piedra  Papel  Tijera
REALIDAD Piedra    45      3      2     ← 45 correctos
         Papel      2     40      8
         Tijera     1      7     42
```

**Interpretación**:
- Diagonal = predicciones correctas
- Fuera de diagonal = errores

**Código**:
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicciones)
print(cm)
```

#### 3. Classification Report

Resumen completo con precision, recall, f1-score:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predicciones))
```

Output:
```
              precision    recall  f1-score   support

      piedra       0.94      0.90      0.92        50
       papel       0.80      0.80      0.80        50
      tijera       0.81      0.84      0.82        50

    accuracy                           0.85       150
```

**¿Qué significa cada métrica?**

Para entenderlo, usa esta matriz de confusión como ejemplo:

```
                PREDICHO
           piedra  papel  tijera
REAL piedra   45      3       2     (50 total)
     papel     2     40       8     (50 total)
     tijera    1      7      42     (50 total)
```

**PRECISION** (Precisión por clase):
> "De todas las veces que predije X, ¿cuántas eran realmente X?"

- **Precision de piedra = 45/(45+2+1) = 45/48 = 0.94**
  - De 48 veces que predije "piedra", 45 eran piedra → 94% correcto
  - "Cuando digo piedra, casi siempre acierto"

- **Precision de papel = 40/(3+40+7) = 40/50 = 0.80**
  - De 50 veces que predije "papel", 40 eran papel → 80% correcto

**RECALL** (Exhaustividad por clase):
> "De todas las X reales, ¿cuántas detecté?"

- **Recall de piedra = 45/(45+3+2) = 45/50 = 0.90**
  - De 50 piedras reales, detecté 45 → 90%
  - "Detecto 9 de cada 10 piedras"

- **Recall de papel = 40/(2+40+8) = 40/50 = 0.80**
  - De 50 papeles reales, detecté 40 → 80%

**F1-SCORE**:
> "Media armónica de precision y recall"

```
F1 = 2 × (precision × recall) / (precision + recall)
```

- **F1 de piedra = 2 × (0.94 × 0.90) / (0.94 + 0.90) = 0.92**
- Balance entre precision y recall
- Útil cuando quieres una única métrica por clase

**SUPPORT**:
> "Cuántas instancias reales hay de cada clase en el test set"

- Support de piedra = 50 → hay 50 piedras en test

**ACCURACY** (al final):
> "Precisión global = (45+40+42)/150 = 0.85"

---

**¿Cuándo importa cada métrica?**

| Métrica | Cuándo importa más |
|---------|-------------------|
| **Precision** | Cuando los **falsos positivos** son malos<br>Ej: Spam → No quiero que emails buenos vayan a spam |
| **Recall** | Cuando los **falsos negativos** son malos<br>Ej: Diagnóstico cáncer → No quiero perder ningún caso |
| **F1-Score** | Cuando quieres balance entre ambos |
| **Accuracy** | Cuando todas las clases importan igual |

**Ejemplo PPT**:
- Si precision de "piedra" es baja → Muchas veces predigo piedra cuando no es
- Si recall de "piedra" es bajo → Me pierdo muchas piedras reales
- En PPT, normalmente nos importa accuracy global (todas las jugadas valen igual)

### Métricas para Regresión

#### Mean Squared Error (MSE)

Promedio de los errores al cuadrado:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predicciones)
print(f"MSE: {mse:.2f}")
```

#### R² Score

Qué tan bien el modelo explica la varianza (0 a 1, más alto = mejor):

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predicciones)
print(f"R² Score: {r2:.2f}")
```

---

## Overfitting y Underfitting

### Underfitting (Subajuste)

**Problema**: El modelo es demasiado simple y no aprende los patrones.

**Síntomas**:
- Mal rendimiento en train
- Mal rendimiento en test

**Analogía**: Estudiante que no estudió para el examen.

```
Train accuracy: 60%
Test accuracy:  58%
```

**Solución**: Usar un modelo más complejo o más features.

### Overfitting (Sobreajuste)

**Problema**: El modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales.

**Síntomas**:
- Excelente rendimiento en train
- Mal rendimiento en test

**Analogía**: Estudiante que memorizó respuestas pero no entendió conceptos.

```
Train accuracy: 99%
Test accuracy:  65%  ← Gran diferencia!
```

**Solución**:
- Usar más datos
- Regularización
- Validación cruzada
- Simplificar modelo

### El Balance Ideal

```
Train accuracy: 85%
Test accuracy:  83%  ← Similar al train, ¡buena señal!
```

### Visualización

```
Error
  ↑
  |     Underfitting    Sweet Spot    Overfitting
  |         ___           _____         /\
  |        /            /     \       /  \____
  |       /            /       \     /
  |______/____________/_________\___/___________→
                Complejidad del modelo
```

---

## Aplicación a Piedra, Papel o Tijera

### Objetivo

Entrenar un modelo que prediga la próxima jugada del oponente basándose en:
- Historial de jugadas
- Patrones detectados
- Features de frecuencia, rachas, etc.

### Features que usaremos

De la Clase 06, ya tenemos features como:
- `freq_global_piedra`, `freq_global_papel`, `freq_global_tijera`
- `lag_1_piedra`, `lag_2_piedra`, etc. (últimas jugadas)
- `racha_victorias`, `racha_derrotas`
- `entropia_5`, `entropia_10`
- `markov_piedra_papel`, etc.

### Target (lo que queremos predecir)

```python
# Columna 'proxima_jugada_oponente'
# Valores posibles: "piedra", "papel", "tijera"
```

### Pipeline completo

```python
# 1. Cargar datos con features ya creadas
df = pd.read_csv('datos_ppt_con_features.csv')

# 2. Preparar X (features) e y (target)
features_a_usar = [
    'freq_global_piedra', 'freq_global_papel', 'freq_global_tijera',
    'lag_1_piedra', 'lag_1_papel', 'lag_1_tijera',
    'racha_victorias', 'entropia_5'
]

X = df[features_a_usar]
y = df['proxima_jugada_oponente']

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrenar modelo
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

# 5. Evaluar
predicciones = modelo.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)

print(f"Accuracy: {accuracy:.2%}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, predicciones))

# 6. Predecir próxima jugada
nueva_ronda = [[0.4, 0.3, 0.3, 1, 0, 0, 2, 1.2]]
proxima_jugada = modelo.predict(nueva_ronda)
print(f"\nPredicción: {proxima_jugada[0]}")
```

### Estrategia de juego

Una vez entrenado el modelo:

```python
def jugar_contra_oponente(modelo, features_actuales):
    # 1. Predecir jugada del oponente
    prediccion = modelo.predict([features_actuales])[0]

    # 2. Jugar el counter
    counter = {
        "piedra": "papel",
        "papel": "tijera",
        "tijera": "piedra"
    }

    return counter[prediccion]
```

---

## Algoritmos Recomendados para PPT

### 1. K-Nearest Neighbors (KNN)

**Idea**: "Dime con quién andas y te diré quién eres"

Busca las 5 situaciones más similares en el historial y predice lo más común.

**Ventajas**:
- Simple de entender
- Funciona bien con patrones locales

**Desventajas**:
- Lento con muchos datos
- Sensible a features irrelevantes

```python
from sklearn.neighbors import KNeighborsClassifier

modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, y_train)
```

### 2. Decision Tree (Árbol de Decisión)

**Idea**: Serie de preguntas tipo "Si-Entonces"

```
¿freq_piedra > 0.4?
├─ SÍ: ¿ultima_jugada = piedra?
│  ├─ SÍ: Predecir PAPEL
│  └─ NO: Predecir PIEDRA
└─ NO: ¿entropia > 1.2?
   └─ ...
```

**Ventajas**:
- Fácil de visualizar
- Captura interacciones entre features

**Desventajas**:
- Tiende a overfitting

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(max_depth=5)
modelo.fit(X_train, y_train)
```

### 3. Random Forest

**Idea**: Muchos árboles votando

Crea 100 árboles diferentes y cada uno vota. La predicción final es la más votada.

**Ventajas**:
- Muy robusto
- Reduce overfitting
- Excelente rendimiento general

**Desventajas**:
- Más lento
- Menos interpretable

```python
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(n_estimators=100, max_depth=5)
modelo.fit(X_train, y_train)
```

---

## Consejos Prácticos

### 1. Empieza simple

No uses el modelo más complejo. Prueba primero:
1. Baseline simple (siempre predecir la jugada más común)
2. KNN
3. Decision Tree
4. Random Forest

### 2. Itera

```
Primera versión: 35% accuracy
↓
Añadir más features: 42% accuracy
↓
Probar otro algoritmo: 48% accuracy
↓
Ajustar hiperparámetros: 52% accuracy
```

### 3. No esperes 100% accuracy

En PPT, incluso humanos profesionales logran ~55-60% contra otros humanos. Si logras >50%, ¡ya estás ganando!

### 4. Valida con datos reales

El modelo debe funcionar contra oponentes NUEVOS, no solo contra el mismo oponente del entrenamiento.

---

## Resumen

| Concepto | Descripción |
|----------|-------------|
| **Entrenamiento** | Mostrar ejemplos al algoritmo para que aprenda |
| **Train/Test Split** | Dividir datos en 80% entrenamiento, 20% prueba |
| **Clasificación** | Predecir categorías (piedra, papel, tijera) |
| **Regresión** | Predecir números continuos (precios, temperatura) |
| **Accuracy** | Porcentaje de predicciones correctas |
| **Overfitting** | Memorizar en lugar de aprender (train bien, test mal) |
| **Underfitting** | Modelo demasiado simple (train mal, test mal) |

---

## Próximos Pasos

1. Practicar con ejemplos simples (iris, diabetes datasets)
2. Aplicar a datos de PPT con features ya creadas
3. Experimentar con diferentes algoritmos
4. Evaluar y comparar resultados
5. Implementar estrategia de juego basada en predicciones

**Clase 08 (próxima)**: Optimización de hiperparámetros, validación cruzada, y despliegue del modelo.
