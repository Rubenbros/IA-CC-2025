# Prompt Engineering

## 1. ¿Qué es un Prompt?

Un **prompt** es la instrucción que le das a un modelo de IA. La calidad de la respuesta depende mucho de cómo preguntes.

**Ejemplo:**
- ❌ Prompt vago: "Háblame de perros"
- ✅ Prompt claro: "Dame 3 razas de perros ideales para apartamentos pequeños, con sus pros y contras"

---

## 2. Estructura de un Buen Prompt

Un prompt efectivo suele tener:

| Elemento | Qué es | Ejemplo |
|----------|--------|---------|
| **Contexto** | Quién eres o situación | "Soy estudiante de medicina" |
| **Tarea** | Qué quieres que haga | "Explícame cómo funciona el corazón" |
| **Formato** | Cómo quieres la respuesta | "En 5 puntos, lenguaje sencillo" |
| **Restricciones** | Qué evitar | "Sin usar términos técnicos" |

**Prompt completo:**
> "Soy estudiante de medicina de primer año. Explícame cómo funciona el sistema circulatorio en 5 puntos clave, usando lenguaje sencillo y sin términos muy técnicos."

---

## 3. Técnicas Básicas

### Zero-Shot (Sin ejemplos)
Pides directamente sin dar ejemplos previos.
> "Clasifica este texto como positivo o negativo: 'La película fue aburrida'"

### Few-Shot (Con ejemplos)
Das ejemplos de lo que quieres antes de pedir.
> "Clasifica estos textos:
> - 'Me encantó' → Positivo
> - 'Qué horror' → Negativo
> - 'La película fue aburrida' → ?"

### Chain of Thought (Paso a paso)
Pides que razone antes de responder. Mejora mucho las respuestas en problemas complejos.
> "Resuelve este problema paso a paso: Si tengo 23 manzanas y regalo 7, luego compro 12 más, ¿cuántas tengo?"

---

## 4. El Truco Mágico: Asignar un Rol

Decirle al modelo "quién es" cambia completamente sus respuestas.

| Rol | Efecto |
|-----|--------|
| "Eres un profesor de primaria" | Explica simple |
| "Eres un experto en marketing" | Usa jerga de marketing |
| "Eres un crítico exigente" | Da opiniones duras |
| "Eres un comediante" | Responde con humor |

**Prueba esto en ChatGPT o Claude:**
> "Eres un chef italiano con 30 años de experiencia. Dame tu opinión sobre poner ketchup en la pasta."

---

## 5. Errores Comunes

| Error | Problema | Solución |
|-------|----------|----------|
| **Muy vago** | "Ayúdame con mi trabajo" | Sé específico: qué trabajo, qué necesitas |
| **Muy largo** | 500 palabras de contexto innecesario | Ve al grano |
| **Sin formato** | No indicas cómo quieres la respuesta | Pide lista, tabla, párrafo, etc. |
| **Asumes que sabe** | "Como te dije antes..." | El modelo no recuerda chats anteriores |

---

## 6. Plantillas Útiles

### Para explicaciones
> "Explícame [TEMA] como si tuviera [EDAD] años, en máximo [N] frases."

### Para resolver problemas
> "Tengo este problema: [PROBLEMA]. Piensa paso a paso y dame una solución."

### Para mejorar textos
> "Mejora este texto haciéndolo más [claro/formal/corto]: [TU TEXTO]"

### Para generar ideas
> "Dame 5 ideas originales para [OBJETIVO]. Para cada idea incluye: nombre, descripción breve, y por qué funcionaría."

---

## 7. Practica

Abre ChatGPT, Claude o Gemini y prueba:

1. **Sin rol vs con rol:**
   - "¿Qué opinas de los videojuegos?"
   - "Eres un psicólogo infantil. ¿Qué opinas de los videojuegos para niños?"

2. **Vago vs específico:**
   - "Escribe un email"
   - "Escribe un email formal pidiendo una extensión de plazo para entregar un proyecto universitario. Tono respetuoso, máximo 100 palabras."

3. **Chain of Thought:**
   - "¿Cuánto es 17 × 24?"
   - "Calcula 17 × 24 paso a paso, mostrando tu razonamiento"

---

## 8. Resumen

- **Prompt** = instrucción que le das a la IA
- **Sé específico**: contexto + tarea + formato
- **Asigna roles**: cambia cómo responde
- **Pide paso a paso**: mejora respuestas complejas
- **Da ejemplos**: si quieres un formato específico
- **Itera**: si no sale bien, reformula

