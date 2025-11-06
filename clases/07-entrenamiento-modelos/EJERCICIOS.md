# Ejercicios Clase 07: Entrenamiento de Modelos

## Orden recomendado

### 1. Ejemplos básicos (para entender conceptos)

Ejecuta estos archivos en orden para familiarizarte con los conceptos:

```bash
cd src/clase07_entrenamiento

# 1. Train/Test Split
python 01_train_test_split.py

# 2. Regresión Lineal
python 02_regresion_lineal.py

# 3. Clasificación (KNN y Decision Trees)
python 03_clasificacion.py
```

### 2. Ejercicio principal: PPT con ML (PARA IMPLEMENTAR)

**Este es TU ejercicio principal. Debes implementarlo desde cero usando tus datos reales.**

El archivo `04_ejercicio_ppt_ALUMNO.py` tiene la estructura básica con TODOs.

**REQUISITO PREVIO**:
- Debes tener un CSV con tus partidas contra compañeros (mínimo 100 rondas)
- Formato necesario:
  ```csv
  numero_ronda,jugada_jugador,jugada_oponente
  1,piedra,papel
  2,tijera,tijera
  3,papel,piedra
  ...
  ```
- Si no tienes suficientes datos: ¡juega más partidas con tus compañeros!

**Tu tarea**:
1. Preparar tu CSV con datos reales de partidas

2. Implementar todas las funciones marcadas con `TODO` en:
   ```bash
   # Abre el archivo y completa los TODOs
   04_ejercicio_ppt_ALUMNO.py
   ```

3. El ejercicio debe:
   - Cargar TUS datos de partidas
   - Generar features (aplicar Clase 06)
   - Entrenar múltiples modelos (KNN, Decision Tree, Random Forest)
   - Comparar resultados
   - Lograr >50% accuracy (mejor que aleatorio 33%)
   - (Bonus) Implementar clase JugadorIA

**Recursos de ayuda**:
- Revisa ejemplos 01-03
- Repasa Clase 06 (Feature Engineering)
- Lee `clases/07-entrenamiento-modelos/README.md`

---

## Desafíos adicionales

### Desafío 1: Mejorar las features

**Objetivo**: Añadir features más avanzadas al modelo de PPT

**Tareas**:
1. Implementar features de entropía (de la clase 06)
2. Añadir features de cadenas de Markov
3. Incluir features de respuesta a resultados
4. Comparar accuracy con features básicas vs avanzadas

**Pistas**:
- Usa `PPTFeatureEngineering` de `src/clase05_fundamentos_ia/feature_engineering_ppt.py`
- Modifica `generar_features_basicas()` en el ejercicio 04

### Desafío 2: Optimizar hiperparámetros

**Objetivo**: Encontrar los mejores hiperparámetros para cada modelo

**Tareas**:
1. Probar diferentes valores de K en KNN (1, 3, 5, 7, 9, 11)
2. Probar diferentes profundidades en Decision Tree (2, 3, 5, 7, 10)
3. Experimentar con Random Forest (n_estimators: 10, 50, 100)
4. Graficar cómo cambia el accuracy con cada hiperparámetro

**Código inicial**:
```python
for k in [1, 3, 5, 7, 9]:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    acc = modelo.score(X_test, y_test)
    print(f"K={k}: {acc:.2%}")
```

### Desafío 3: Validación cruzada

**Objetivo**: Evaluar modelos de forma más robusta

**Tareas**:
1. Implementar 5-fold cross-validation
2. Comparar accuracy promedio vs un solo train/test split
3. Calcular desviación estándar del accuracy

**Pistas**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modelo, X, y, cv=5)
print(f"Accuracy promedio: {scores.mean():.2%} ± {scores.std():.2%}")
```

### Desafío 4: Ensemble de modelos

**Objetivo**: Combinar predicciones de múltiples modelos

**Tareas**:
1. Entrenar KNN, Decision Tree y Random Forest
2. Hacer cada modelo prediga la próxima jugada
3. Usar votación mayoritaria para decisión final
4. Comparar con modelos individuales

**Estrategia**:
```python
pred_knn = modelo_knn.predict(X_test)
pred_tree = modelo_tree.predict(X_test)
pred_rf = modelo_rf.predict(X_test)

# Votación: la jugada más predicha
predicciones_ensemble = []
for i in range(len(X_test)):
    votos = [pred_knn[i], pred_tree[i], pred_rf[i]]
    prediccion = max(set(votos), key=votos.count)
    predicciones_ensemble.append(prediccion)
```

### Desafío 5: Diferentes tipos de oponentes

**Objetivo**: Evaluar el modelo contra diferentes estrategias

**Tareas**:
1. Generar datasets con diferentes oponentes:
   ```bash
   python generar_datos_ppt.py
   ```
2. Entrenar modelo con cada tipo de oponente
3. Probar modelo entrenado con un tipo contra otros tipos
4. Analizar qué features son más importantes para cada oponente

**Tipos de oponentes**:
- `basico`: Preferencias fijas (40% piedra, 35% papel, 25% tijera)
- `adaptativo`: Cambia según gana/pierde
- `patrones`: Sigue secuencias repetitivas

### Desafío 6: Visualización de resultados

**Objetivo**: Crear visualizaciones para entender el modelo

**Tareas**:
1. Graficar confusion matrix como heatmap
2. Crear curva de aprendizaje (accuracy vs cantidad de datos)
3. Visualizar importancia de features (para Decision Tree/RF)
4. Graficar evolución del win rate durante la partida

**Código para heatmap**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['piedra', 'papel', 'tijera'],
            yticklabels=['piedra', 'papel', 'tijera'])
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.title('Confusion Matrix')
plt.show()
```

### Desafío 7: Implementar un jugador interactivo

**Objetivo**: Jugar contra tu propio modelo entrenado

**Tareas**:
1. Crear un script interactivo que:
   - Pida tu jugada
   - Prediga con el modelo
   - Muestre el resultado
   - Actualice el historial
2. Jugar 20 rondas contra el modelo
3. Ver si puedes ganarle

**Código inicial**:
```python
jugador_ia = JugadorIA(modelo, feature_gen)

for ronda in range(20):
    # Usuario juega
    tu_jugada = input("Tu jugada (piedra/papel/tijera): ").lower()

    # IA predice y juega
    jugada_ia = jugador_ia.predecir_y_jugar()

    # Mostrar resultado
    print(f"IA jugó: {jugada_ia}")

    # Calcular ganador
    # ...

    # Registrar
    jugador_ia.registrar_resultado(jugada_ia, tu_jugada)
```

---

## Preguntas de reflexión

1. **¿Por qué es importante dividir en train y test?**
   - ¿Qué pasaría si evaluamos con los mismos datos de entrenamiento?

2. **¿Qué significa overfitting?**
   - ¿Cómo lo detectas en tus resultados?
   - ¿Cómo lo solucionarías?

3. **¿Por qué KNN es sensible a la escala de las features?**
   - ¿Cuándo necesitarías normalizar?

4. **¿Cuál modelo es mejor para PPT: KNN o Decision Tree?**
   - ¿Por qué?
   - ¿Depende del tipo de oponente?

5. **¿Es posible tener 100% accuracy en PPT?**
   - ¿Por qué sí o por qué no?
   - ¿Qué accuracy es realista esperar?

6. **¿Cómo afecta la cantidad de datos al rendimiento?**
   - Prueba con 50, 100, 200 rondas
   - ¿Cuál es el mínimo necesario?

---

## Recursos adicionales

### Documentación de scikit-learn

- [Train/Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Datasets de práctica

Los siguientes CSVs están disponibles en `src/clase07_entrenamiento/`:

1. `datos_ppt_ejemplo.csv` - Dataset básico (usado en ejercicios)
2. `datos_ppt_basico.csv` - Oponente con preferencias fijas
3. `datos_ppt_adaptativo.csv` - Oponente adaptativo
4. `datos_ppt_patrones.csv` - Oponente con patrones

Genera más con:
```bash
python generar_datos_ppt.py
```

---

## Evaluación

Tu ejercicio estará completo cuando:

- [ ] Entiendes la diferencia entre train y test
- [ ] Puedes entrenar un modelo KNN
- [ ] Puedes entrenar un Decision Tree
- [ ] Sabes interpretar accuracy y confusion matrix
- [ ] Puedes comparar múltiples modelos
- [ ] Implementas un modelo para PPT con >50% accuracy
- [ ] Entiendes overfitting y underfitting
- [ ] Puedes explicar cuándo usar cada algoritmo

**Bonus**: Completa al menos 2 desafíos adicionales.

---

## Siguiente clase

En la **Clase 08** veremos:
- Optimización de hiperparámetros (GridSearch, RandomSearch)
- Validación cruzada (Cross-Validation)
- Feature importance y selección de features
- Ensembles avanzados
- Despliegue del modelo en producción
