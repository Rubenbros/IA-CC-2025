# Clase 05: Fundamentos de Inteligencia Artificial y Machine Learning

## Descripción General

Esta clase cubre los conceptos fundamentales de Inteligencia Artificial y Machine Learning desde una perspectiva teórica y práctica, con ejemplos aplicables a diversos dominios.

## Contenido del Curso

### 📚 Documentación Teórica

#### 1. [Teoría de Fundamentos de IA y ML](teoria_fundamentos_ia_ml.md)

Documento principal con teoría completa y generalizada:

**Contenido:**
- **Parte 1**: Introducción a la Inteligencia Artificial
  - ¿Qué es IA? Niveles de IA (Débil, Fuerte, Super)
  - Diferencia entre programación tradicional y ML

- **Parte 2**: Fundamentos Matemáticos
  - Probabilidad básica y condicional
  - Teorema de Bayes con ejemplos aplicados
  - Distribuciones de probabilidad (uniforme, normal, binomial)
  - Valor esperado
  - Entropía de Shannon y teoría de la información

- **Parte 3**: Tipos de Aprendizaje Automático
  - Aprendizaje Supervisado (regresión y clasificación)
  - Aprendizaje No Supervisado (clustering, reducción dimensional)
  - Aprendizaje por Refuerzo (Q-Learning, políticas)
  - Comparación y cuándo usar cada tipo

- **Parte 4**: Algoritmos Fundamentales
  - Regresión Lineal
  - Cadenas de Markov
  - N-gramas y modelos de lenguaje
  - K-Nearest Neighbors (KNN)
  - Ensemble Methods (Bagging, Boosting, Stacking)

- **Parte 5**: Métricas y Evaluación
  - Métricas de clasificación (Accuracy, Precision, Recall, F1)
  - Matriz de confusión
  - Métricas de regresión (MSE, RMSE, R², MAE)
  - Train/Validation/Test split
  - K-Fold Cross-Validation
  - Overfitting vs Underfitting

- **Parte 6**: Conceptos Avanzados
  - Teoría de la información
  - Batch vs Online Learning
  - Pipeline de ML
  - Feature Engineering

#### 2. [Ejemplos y Aplicaciones Prácticas](ejemplos_aplicaciones_ia.md)

Documento con casos de uso reales y aplicaciones en diversos dominios:

**Contenido:**
- **Procesamiento de Lenguaje Natural (NLP)**
  - Análisis de sentimientos
  - Chatbots y asistentes virtuales
  - Clasificación de intenciones

- **Visión por Computadora**
  - Clasificación de imágenes (MNIST, ImageNet)
  - Detección de objetos (YOLO, R-CNN)
  - Segmentación semántica (U-Net)

- **Sistemas de Recomendación**
  - Filtrado colaborativo (user-based, item-based)
  - Filtrado basado en contenido
  - Sistemas híbridos (Netflix)

- **Detección de Fraude**
  - Fraude en transacciones bancarias
  - Detección de bots y spam
  - Análisis de anomalías

- **Predicción de Series Temporales**
  - Predicción de demanda (ARIMA, Prophet)
  - Forecasting de ventas
  - Predicción de precios

- **Diagnóstico Médico Asistido**
  - Detección de cáncer en imágenes
  - Predicción de readmisión hospitalaria
  - Transfer learning en medicina

#### 3. [Feature Engineering: Teoría y Práctica](feature_engineering_teoria.md) ⭐ NUEVO

Documento completo sobre ingeniería de características aplicada al proyecto PPT:

**Contenido:**
- **Fundamentos Teóricos**
  - ¿Qué es Feature Engineering y por qué es crítico?
  - El problema de representación y espacio de features
  - Información mutua y maldición de la dimensionalidad
  - Principio de parsimonia (Navaja de Occam)

- **Tipos de Features**
  - Features básicas (directas)
  - Features derivadas (transformaciones)
  - Features de agregación (estadísticos)
  - Features temporales (ventanas, lags)
  - Features de codificación (Label, One-Hot)
  - Features de interacción (combinaciones)
  - Features de dominio (conocimiento experto)

- **Técnicas Avanzadas**
  - Extracción de componentes
  - Binning (discretización)
  - Transformaciones matemáticas (log, sqrt, normalización)
  - Ventanas deslizantes (sliding windows)
  - Lag features (retardos)
  - Features de frecuencia
  - Features de entropía (aleatoriedad)

- **Aplicación Específica a Piedra, Papel o Tijera**
  - 8 categorías de features para el proyecto
  - Ejemplo completo: de datos crudos a vector de features
  - Código de implementación en Python
  - Visualizaciones y análisis

- **Validación y Selección**
  - Análisis de correlación
  - Información mutua
  - Feature importance
  - Eliminación de features redundantes
  - Validación temporal (crítica para secuencias)

- **Mejores Prácticas**
  - Evitar data leakage
  - Escalado de features
  - Manejo de valores faltantes
  - Feature engineering iterativo
  - Documentación

- **Ejercicios Propuestos**
  - 6 ejercicios prácticos con soluciones

### 💻 Notebooks Prácticos

#### 1. [Ejemplos ML Generales](../../src/clase05_fundamentos_ia/ejemplos_ml_generales.ipynb)

Notebook interactivo con código ejecutable:

**Contenido:**
- **Regresión Lineal**: Predicción de precios de casas
  - Feature engineering
  - Interpretación de coeficientes
  - Métricas (R², RMSE)
  - Visualización de predicciones

- **Clasificación**: Detección de spam en emails
  - Vectorización de texto (TF-IDF)
  - Naive Bayes classifier
  - Matriz de confusión
  - Precision vs Recall

- **Clustering**: Segmentación de clientes
  - K-Means clustering
  - Método del codo para elegir K
  - Normalización de datos
  - Interpretación de clusters

#### 2. [Teoría Completa con Código](../../src/clase05_fundamentos_ia/teoria_completa_con_codigo.ipynb)

Notebook original con ejemplos aplicados (incluye caso de estudio de juegos estratégicos).

#### 3. [Feature Engineering PPT](../../src/clase05_fundamentos_ia/feature_engineering_ppt.py) ⭐ NUEVO

Script Python con implementación completa de feature engineering para el proyecto Piedra, Papel o Tijera:

**Contenido:**
- **Clase PPTFeatureEngineering**: Implementación completa y reutilizable
- **Features de Frecuencia**: Globales y en ventanas temporales
- **Features de Patrones**: Lags, bigramas, trigramas, detección de cambios
- **Features de Rachas**: Victorias/derrotas consecutivas, récords
- **Features Temporales**: Tiempo de reacción, fases del juego, aceleración
- **Features de Entropía**: Medición de aleatoriedad y predictibilidad
- **Features de Markov**: Matrices de transición, predicciones probabilísticas
- **Features de Respuesta**: Cómo reacciona el oponente a victorias/derrotas
- **Ejemplos de Uso**: 3 ejemplos completos con visualizaciones
- **Código Documentado**: Listo para usar en el proyecto

**Cómo ejecutar:**
```bash
python src/clase05_fundamentos_ia/feature_engineering_ppt.py
```

#### 4. [Ejemplo de Uso con CSV](../../src/clase05_fundamentos_ia/ejemplo_uso_csv.py) ⭐ NUEVO

Script completo que muestra el workflow desde CSV hasta dataset listo para ML:

**Contenido:**
- **Cargar CSV básico**: Solo 3 columnas (numero_ronda, jugada_jugador, jugada_oponente)
- **Generar features para cada ronda**: Procesamiento completo del historial
- **Crear DataFrame final**: Más de 30 features generadas automáticamente
- **Preparar para ML**: Separación de features (X) y objetivo (y)
- **Ejemplo incremental**: Cómo usar en tiempo real durante el juego
- **Código comentado**: Cada paso explicado claramente

**Cómo ejecutar:**
```bash
python src/clase05_fundamentos_ia/ejemplo_uso_csv.py
```

**Resultado:** Genera dos archivos CSV:
- `dataset_ppt_ejemplo.csv` - Datos originales (3 columnas)
- `dataset_ppt_con_features.csv` - Dataset completo para entrenar modelos (30+ columnas)

## Cómo Usar Este Material

### Para Estudiantes

**Ruta de Aprendizaje Sugerida:**

1. **Empezar con teoría básica** (1-2 horas):
   - Leer Parte 1 y 2 de `teoria_fundamentos_ia_ml.md`
   - Entender conceptos de probabilidad y entropía

2. **Tipos de aprendizaje** (1 hora):
   - Leer Parte 3 de `teoria_fundamentos_ia_ml.md`
   - Entender cuándo usar cada tipo

3. **Práctica con ejemplos** (2-3 horas):
   - Ejecutar `ejemplos_ml_generales.ipynb`
   - Experimentar cambiando parámetros
   - Probar con datos propios

4. **Algoritmos específicos** (2 horas):
   - Leer Parte 4 de `teoria_fundamentos_ia_ml.md`
   - Implementar versiones simples de algoritmos

5. **Métricas y evaluación** (1 hora):
   - Leer Parte 5
   - Practicar interpretación de matrices de confusión

6. **Feature Engineering** (3-4 horas) ⭐ NUEVO:
   - Leer `feature_engineering_teoria.md` completo
   - Ejecutar `feature_engineering_ppt.py`
   - Hacer los ejercicios propuestos
   - Aplicar al proyecto PPT

7. **Aplicaciones reales** (1-2 horas):
   - Leer `ejemplos_aplicaciones_ia.md`
   - Elegir un dominio de interés
   - Investigar más sobre ese dominio

### Para Docentes

**Material didáctico incluido:**

- **Diapositivas potenciales**: Usar secciones de `teoria_fundamentos_ia_ml.md`
- **Ejercicios prácticos**: Adaptar notebooks para laboratorios
- **Evaluaciones**: Usar conceptos teóricos y ejercicios prácticos
- **Proyectos**: Inspirarse en `ejemplos_aplicaciones_ia.md`

**Estructura de clase sugerida:**

- **Sesión 1** (2h): Introducción a IA, probabilidad básica
- **Sesión 2** (2h): Tipos de aprendizaje, algoritmos básicos
- **Sesión 3** (2h): Práctica con regresión y clasificación
- **Sesión 4** (2h): Clustering y métricas avanzadas
- **Sesión 5** (2h): Aplicaciones reales y proyecto final

## Recursos Adicionales

### Libros Recomendados

**Nivel Principiante:**
- "The Hundred-Page Machine Learning Book" - Andriy Burkov
- "Machine Learning for Absolute Beginners" - Oliver Theobald

**Nivel Intermedio:**
- "Hands-On Machine Learning" - Aurélien Géron
- "Python Machine Learning" - Sebastian Raschka

**Nivel Avanzado:**
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

### Cursos Online

- **Andrew Ng - Machine Learning** (Coursera)
- **Fast.ai - Practical Deep Learning**
- **Google Machine Learning Crash Course**
- **MIT 6.034 Artificial Intelligence** (YouTube)

### Datasets para Practicar

- **Kaggle**: Miles de datasets y competiciones
- **UCI ML Repository**: Datasets clásicos educativos
- **Scikit-learn**: Datasets integrados para práctica rápida
- **TensorFlow Datasets**: Colección amplia para deep learning

### Herramientas y Librerías

**Python Essentials:**
```python
# Datos y computación
import pandas as pd
import numpy as np

# Machine Learning
from sklearn import *
import xgboost as xgb
import lightgbm as lgb

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
import torch
```

## Proyectos Sugeridos

### Nivel Básico (5-6 puntos)

1. **Predictor de precios**: Regresión lineal para precios de casas/coches
2. **Clasificador de flores**: Iris dataset con múltiples algoritmos
3. **Análisis exploratorio**: Dataset de tu elección con visualizaciones

### Nivel Intermedio (7-8 puntos)

4. **Sistema de recomendación**: Películas, música o productos
5. **Detector de spam**: Con features personalizadas
6. **Segmentación de clientes**: K-Means + análisis de negocios
7. **Predicción de churn**: Clasificación con datos desbalanceados

### Nivel Avanzado (9-10 puntos)

8. **Análisis de sentimientos**: NLP con redes neuronales
9. **Clasificador de imágenes**: CNN con transfer learning
10. **Sistema de detección de fraude**: Anomaly detection + ensemble
11. **Chatbot simple**: Intent classification + response generation
12. **Predicción de series temporales**: ARIMA o LSTM

## Evaluación

### Criterios de Evaluación

**Conocimientos Teóricos (40%):**
- Comprensión de conceptos fundamentales
- Capacidad de explicar algoritmos
- Entendimiento de métricas

**Implementación Práctica (40%):**
- Código funcional y bien estructurado
- Uso apropiado de librerías
- Validación correcta de modelos

**Análisis y Comunicación (20%):**
- Interpretación de resultados
- Visualizaciones claras
- Documentación del proceso

### Rúbrica de Proyecto

| Aspecto | Básico (5-6) | Intermedio (7-8) | Avanzado (9-10) |
|---------|--------------|------------------|-----------------|
| **Complejidad** | Algoritmo simple, datos limpios | Múltiples algoritmos, preprocesamiento | Ensemble, feature engineering avanzado |
| **Métricas** | Accuracy básico | Múltiples métricas, validación cruzada | Análisis profundo, intervalos confianza |
| **Código** | Funcional, básico | Modular, comentado | Producción-ready, tests |
| **Análisis** | Descripción básica | Interpretación detallada | Insights accionables, recomendaciones |

## Preguntas Frecuentes

**P: ¿Necesito conocimientos avanzados de matemáticas?**
R: No necesariamente. Los conceptos básicos de probabilidad y álgebra lineal son suficientes para empezar.

**P: ¿Qué lenguaje de programación es mejor?**
R: Python es el estándar de facto en ML. R también es popular en estadística.

**P: ¿Cuánto tiempo toma aprender ML?**
R: Conceptos básicos: 2-3 meses. Nivel intermedio: 6-12 meses. Maestría: años de práctica.

**P: ¿Necesito GPU para entrenar modelos?**
R: Para empezar, no. CPU es suficiente. Para deep learning, GPU acelera significativamente.

**P: ¿Cómo encuentro mi primer trabajo en ML?**
R: Portfolio en GitHub, proyectos en Kaggle, contribuciones open source, networking.

## Contribuciones

Este material es educativo y está en constante evolución. Sugerencias y mejoras son bienvenidas.

## Licencia

Material educativo para uso académico.

---

**Última actualización**: Octubre 2024

**Autor**: Curso IA-CC-2025

**Contacto**: [Detalles del instructor]
