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
| GPT-4 | OpenAI | ChatGPT (limitado) |
| Claude | Anthropic | claude.ai |
| Gemini | Google | gemini.google.com |
| Llama 3 | Meta | Groq, Ollama |

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

