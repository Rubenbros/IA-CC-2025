# RAG (Retrieval-Augmented Generation)

## 1. ¿Qué es RAG?

**RAG** = Generación Aumentada por Recuperación

Es una técnica que permite a la IA responder usando TUS documentos, no solo lo que aprendió en su entrenamiento.

**El problema:**
- ChatGPT no conoce los documentos de tu empresa
- No tiene información actualizada de hoy
- No puede acceder a tus PDFs o bases de datos

**La solución RAG:**
1. Guardas tus documentos en una base de datos especial
2. Cuando preguntas algo, busca documentos relevantes
3. Le pasa esos documentos a la IA junto con tu pregunta
4. La IA responde basándose en TUS documentos

---

## 2. Cómo Funciona (Simplificado)

```
TÚ: "¿Cuál es la política de vacaciones de la empresa?"

SISTEMA RAG:
1. Busca en tus documentos → Encuentra "manual_empleados.pdf"
2. Extrae la sección relevante
3. Le dice al LLM: "Basándote en esto, responde la pregunta"

IA: "Según el manual de empleados, tienes 22 días de vacaciones..."
```

---

## 3. Conceptos Clave

### Embeddings
Convertir texto en números que representan su significado.
- Textos parecidos → números parecidos
- Permite buscar por significado, no solo palabras exactas

### Base de Datos Vectorial
Donde se guardan esos números (embeddings).
- Optimizada para buscar "textos similares"
- Ejemplos: ChromaDB, Pinecone, FAISS

### Chunking
Dividir documentos largos en trozos pequeños.
- Un PDF de 100 páginas → 500 fragmentos pequeños
- Permite buscar partes específicas, no todo el documento

---

## 4. El Proceso Paso a Paso

### Fase 1: Preparación (una vez)
```
Tus documentos → Dividir en trozos → Convertir a embeddings → Guardar en base de datos
```

### Fase 2: Consulta (cada pregunta)
```
Tu pregunta → Buscar trozos relevantes → Juntar con la pregunta → Enviar al LLM → Respuesta
```

---

## 5. Herramientas Gratuitas para RAG

| Herramienta | Tipo | Descripción |
|-------------|------|-------------|
| **ChromaDB** | Base de datos | Simple, gratuita, local |
| **FAISS** | Base de datos | Muy rápida (de Facebook) |
| **LangChain** | Framework | Facilita crear pipelines RAG |
| **all-MiniLM-L6-v2** | Modelo embeddings | Gratuito, buena calidad |

---

## 6. Casos de Uso

| Uso | Ejemplo |
|-----|---------|
| **Chatbot de soporte** | Responde sobre tu producto usando la documentación |
| **Asistente legal** | Busca en contratos y leyes |
| **Tutor personal** | Responde sobre los apuntes del curso |
| **Búsqueda empresarial** | Encuentra info en documentos internos |

---

## 7. Ventajas y Limitaciones

**Ventajas:**
- La IA usa TUS datos, no inventa
- Información siempre actualizada
- Privacidad: tus datos no van a entrenar modelos
- Reduce alucinaciones

**Limitaciones:**
- Requiere preparar los documentos
- La búsqueda puede fallar
- Más lento que preguntar directamente
- Requiere mantenimiento

---

## 8. RAG que Ya Puedes Usar

Algunos servicios ya tienen RAG integrado:

| Servicio | Cómo usarlo |
|----------|-------------|
| **ChatGPT** | Sube archivos y pregunta sobre ellos |
| **Claude** | Sube PDFs (hasta 200k tokens de contexto) |
| **Perplexity** | Busca en internet automáticamente |
| **NotebookLM** (Google) | Crea un "cuaderno" con tus documentos |

**Prueba:** Sube un PDF a Claude y pregunta algo específico sobre él.

---

## 9. Resumen

- **RAG** = hacer que la IA responda usando tus documentos
- **Embeddings** = representar texto como números para buscar por significado
- **Base vectorial** = donde se guardan los embeddings
- **Ya lo usas**: cuando subes un PDF a ChatGPT o Claude
- **Útil para**: soporte, documentación, búsqueda interna

