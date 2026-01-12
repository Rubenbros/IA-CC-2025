# Fundamentos de IA Generativa

## 1. ¿Qué es la IA Generativa?

La **IA Generativa** crea contenido nuevo: texto, imágenes, código, música. No solo analiza datos, los genera.

**Ejemplos que puedes probar gratis:**
- **ChatGPT** (chat.openai.com) - El más conocido
- **Claude** (claude.ai) - Muy bueno para textos largos y código
- **Gemini** (gemini.google.com) - De Google, integrado con sus servicios
- **Copilot** (copilot.microsoft.com) - De Microsoft, basado en GPT

---

## 2. ¿Qué son los LLMs?

**LLM = Large Language Model** (Modelo Grande de Lenguaje)

Son programas entrenados con cantidades enormes de texto (internet, libros, artículos) que aprenden a predecir "qué palabra viene después". Esto les permite:
- Responder preguntas
- Escribir textos
- Traducir idiomas
- Generar código
- Razonar sobre problemas

**Modelos populares:**
| Modelo | Creador | Acceso Gratuito |
|--------|---------|-----------------|
| GPT-4o | OpenAI | ChatGPT (limitado) |
| Claude 3.5 | Anthropic | claude.ai |
| Gemini 2.0 | Google | gemini.google.com |
| Llama 3.3 | Meta | Groq, Ollama |
| DeepSeek V3 | DeepSeek | chat.deepseek.com |

---

## 2.1. Breve Historia de los LLMs

| Año | Hito | Importancia |
|-----|------|-------------|
| **2017** | Paper "Attention is All You Need" | Nace la arquitectura **Transformer** |
| **2018** | GPT-1 (OpenAI), BERT (Google) | Primeros modelos transformer para lenguaje |
| **2019** | GPT-2 | OpenAI no lo publica por "peligroso" (hoy parece pequeño) |
| **2020** | GPT-3 (175B parámetros) | Demuestra que escala = capacidad emergente |
| **2022** | ChatGPT | Explosión mainstream, 100M usuarios en 2 meses |
| **2023** | GPT-4, Claude 2, Llama 2 | Competencia intensa, modelos open source |
| **2024** | Claude 3, GPT-4o, Llama 3, Gemini 2 | Multimodalidad, agentes, modelos más eficientes |
| **2025** | Agentes autónomos, razonamiento avanzado | Modelos que "piensan" antes de responder |

**La lección clave:** Los LLMs mejoran exponencialmente cuando se les da más datos, más parámetros y mejor entrenamiento.

---

## 2.2. Arquitectura Transformer (Simplificado)

Los LLMs modernos usan la arquitectura **Transformer**. No necesitas entender las matemáticas, pero sí la idea:

### El Mecanismo de Atención
La innovación clave es la **atención** (attention): el modelo puede "mirar" todas las palabras anteriores para decidir qué palabra viene después.

```
Frase: "El gato se subió al _____"

Atención alta: "gato", "subió"
Atención baja: "El", "se"

El modelo presta más atención a las palabras importantes para predecir: "árbol", "tejado", "sofá"
```

### ¿Por qué funciona tan bien?
1. **Paralelización**: A diferencia de modelos anteriores (RNNs), puede procesar toda la frase a la vez
2. **Contexto largo**: Puede "recordar" información de muy atrás en el texto
3. **Escala**: Más capas y parámetros = mejor rendimiento

### Tamaño de los Modelos
| Modelo | Parámetros | Tamaño aproximado |
|--------|------------|-------------------|
| GPT-2 | 1.5B | Pequeño |
| Llama 3 8B | 8B | Mediano (corre en tu PC) |
| GPT-3.5 | ~20B | Grande |
| Llama 3 70B | 70B | Muy grande |
| GPT-4 | ~1.7T (estimado) | Enorme |

*B = Billones (mil millones), T = Trillones*

---

## 2.3. Cómo se Entrenan los LLMs

El entrenamiento tiene **3 fases**:

### Fase 1: Pre-entrenamiento
- Se alimenta al modelo con **cantidades enormes de texto** (internet, libros, código)
- Aprende a predecir la siguiente palabra
- Dura semanas/meses en miles de GPUs
- Coste: millones de dólares

### Fase 2: Fine-tuning (Ajuste fino)
- Se entrena con datos específicos de mayor calidad
- Ejemplos de conversaciones bien escritas
- El modelo aprende el "formato" de respuesta esperado

### Fase 3: RLHF (Aprendizaje por Refuerzo con Feedback Humano)
- Humanos evalúan respuestas del modelo
- El modelo aprende qué respuestas prefieren los humanos
- Esto hace al modelo más útil, seguro y alineado

```
Pre-entrenamiento → Sabe mucho pero es caótico
Fine-tuning → Sabe responder en formato chat
RLHF → Sabe dar respuestas útiles y seguras
```

---

## 2.4. Tipos de Modelos

### Modelos Base vs Modelos Chat

| Modelo Base | Modelo Chat/Instruct |
|-------------|---------------------|
| Solo predice texto | Optimizado para conversación |
| Completa lo que escribes | Responde a instrucciones |
| Difícil de usar directamente | Fácil de usar |
| Ejemplo: Llama 3 Base | Ejemplo: Llama 3 Instruct |

**Ejemplo práctico:**
- **Modelo Base**: Si escribes "La capital de Francia es", responde "París. La capital de España es Madrid. La capital de..."
- **Modelo Chat**: Si preguntas "¿Cuál es la capital de Francia?", responde "La capital de Francia es París."

### Modelos Cerrados vs Open Source

| Cerrados | Open Source |
|----------|-------------|
| GPT-4, Claude, Gemini | Llama 3, Mistral, Qwen |
| Solo por API/web | Puedes descargarlos |
| Más potentes (generalmente) | Puedes modificarlos |
| Dependes de la empresa | Control total |
| Datos van a sus servidores | Privacidad total |

**¿Cuál usar?**
- **Para aprender**: Da igual, usa el que te resulte cómodo
- **Para empresa con datos sensibles**: Open source (local)
- **Para máxima calidad**: Modelos cerrados (por ahora)

---

## 2.5. Multimodalidad

Los LLMs modernos no solo procesan texto, son **multimodales**:

| Capacidad | Modelos que lo tienen |
|-----------|----------------------|
| **Entender imágenes** | GPT-4o, Claude 3, Gemini |
| **Entender audio** | GPT-4o, Gemini |
| **Generar imágenes** | DALL-E 3 (ChatGPT), Midjourney |
| **Entender video** | Gemini 2.0 |
| **Entender código** | Todos los principales |

**Ejemplo práctico:**
> Sube una foto de un error de tu código a Claude o ChatGPT y pregunta "¿Qué está mal aquí?"

**La tendencia:** Los modelos del futuro serán completamente multimodales - entenderán y generarán cualquier tipo de contenido

---

## 3. Conceptos Básicos

### Tokens
Los LLMs no leen palabras, leen **tokens** (trozos de palabras).
- "Hola" = 1 token
- "inteligencia" = 2-3 tokens
- Una página de texto ≈ 500 tokens

### Ventana de Contexto
Cuánto texto puede "recordar" el modelo en una conversación.
- GPT-4: ~128,000 tokens (un libro entero)
- Claude: ~200,000 tokens
- Modelos pequeños: ~4,000 tokens

### Temperature (Creatividad)
Controla qué tan "creativo" o "seguro" es el modelo:
- **0.0** = Siempre la misma respuesta, muy predecible
- **0.7** = Balance (lo más común)
- **1.0** = Muy creativo, puede ser impredecible

---

## 4. Capacidades y Limitaciones

### Lo que SÍ pueden hacer
- Escribir y resumir textos
- Responder preguntas sobre conocimiento general
- Generar y explicar código
- Traducir idiomas
- Ayudar con ideas y brainstorming

### Lo que NO pueden hacer bien
- **Inventan cosas** ("alucinan") - pueden decir datos falsos con confianza
- **No saben la fecha actual** - su conocimiento tiene fecha de corte
- **No navegan internet** (sin herramientas) - solo saben lo que aprendieron
- **Matemáticas complejas** - pueden equivocarse en cálculos

---

## 5. Cómo Funciona una Conversación

Cuando hablas con ChatGPT o Claude:

```
TÚ: "¿Qué es Python?"

[El modelo recibe tu mensaje + instrucciones del sistema]
[Genera la respuesta palabra por palabra]

MODELO: "Python es un lenguaje de programación..."
```

El modelo **no recuerda** conversaciones anteriores. Cada vez que hablas, le envían toda la conversación de nuevo.

---

## 6. Practica: Prueba Estos Agentes

| Herramienta | URL | Para qué es bueno |
|-------------|-----|-------------------|
| **ChatGPT** | chat.openai.com | Todo uso general |
| **Claude** | claude.ai | Textos largos, análisis, código |
| **Gemini** | gemini.google.com | Integración con Google |
| **Perplexity** | perplexity.ai | Búsqueda con fuentes |
| **Poe** | poe.com | Acceso a varios modelos |

**Ejercicio:** Abre Claude o ChatGPT y pregúntale:
1. "Explícame qué eres en 3 frases"
2. "¿Cuál es la fecha de hoy?" (verás que puede equivocarse)
3. "Inventa una receta con los ingredientes: pollo, limón, ajo"

---

## 7. Resumen

- **IA Generativa**: crea contenido nuevo (texto, imágenes, código)
- **LLM**: modelo entrenado con texto que predice palabras
- **Tokens**: unidades que procesa el modelo
- **Temperature**: controla creatividad
- **Limitaciones**: alucinan, no tienen información actualizada
- **Puedes probar gratis**: ChatGPT, Claude, Gemini, Perplexity

