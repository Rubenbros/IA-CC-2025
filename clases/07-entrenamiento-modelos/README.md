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

### 1. Regresión (predecir números)

**Cuándo usar**: Cuando quieres predecir un valor numérico continuo.

**Ejemplos**:
- Predecir precio de una casa (100,000€, 250,000€, etc.)
- Predecir temperatura mañana (18.5°C, 22.3°C, etc.)
- Predecir ventas del mes próximo (1250 unidades)

**Algoritmos comunes**:
- Regresión Lineal
- Regresión Polinomial
- Random Forest Regressor

### 2. Clasificación (predecir categorías)

**Cuándo usar**: Cuando quieres predecir una categoría o clase.

**Ejemplos**:
- Email es spam o no spam (2 clases)
- Reconocer dígitos 0-9 (10 clases)
- **Predecir jugada en PPT** (3 clases: piedra, papel, tijera)

**Algoritmos comunes**:
- K-Nearest Neighbors (KNN)
- Decision Trees (Árboles de Decisión)
- Random Forest
- Logistic Regression
- Support Vector Machines (SVM)

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
