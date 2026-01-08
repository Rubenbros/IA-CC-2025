# Agentes de IA

## 1. ¿Qué es un Agente de IA?

Un **agente** es una IA que puede hacer cosas por ti, no solo responder preguntas.

| Chatbot Normal | Agente |
|----------------|--------|
| Solo responde preguntas | Ejecuta acciones |
| Una respuesta y listo | Múltiples pasos hasta completar |
| No usa herramientas | Usa herramientas (buscar, calcular, enviar emails...) |
| Tú decides todo | Él decide qué hacer |

**Ejemplo práctico:**
- **Chatbot**: "El vuelo más barato a Madrid cuesta unos 50€" (te dice información)
- **Agente**: Busca vuelos → compara precios → te muestra opciones → puede reservar por ti

---

## 2. Agentes que Puedes Usar Gratis

| Agente | URL | Qué puede hacer |
|--------|-----|-----------------|
| **ChatGPT** | chat.openai.com | Buscar en web, analizar datos, generar imágenes, ejecutar código |
| **Claude** | claude.ai | Analizar documentos largos, código, proyectos |
| **Gemini** | gemini.google.com | Buscar en web, integrar con Google |
| **Copilot** | copilot.microsoft.com | Buscar, crear imágenes, integrar con Office |
| **Perplexity** | perplexity.ai | Búsqueda con fuentes citadas |

**Prueba:** Sube un PDF a Claude y pídele que lo resuma → Está usando herramientas internamente.

---

## 3. Cómo Funciona un Agente

El agente sigue un ciclo:

```
1. RECIBE tarea del usuario
2. PIENSA qué necesita hacer
3. EJECUTA una acción (usar herramienta)
4. OBSERVA el resultado
5. REPITE hasta completar la tarea
6. RESPONDE al usuario
```

**Ejemplo: "¿Qué tiempo hará mañana en Madrid?"**

```
PENSAMIENTO: Necesito buscar el pronóstico del tiempo
ACCIÓN: Usar herramienta de búsqueda web
RESULTADO: "Mañana en Madrid: 22°C, soleado"
PENSAMIENTO: Ya tengo la información
RESPUESTA: "Mañana en Madrid hará 22°C y estará soleado"
```

---

## 4. Las Herramientas (Tools)

Las herramientas son las "habilidades" del agente. Sin ellas, solo puede hablar.

| Herramienta | Para qué sirve |
|-------------|----------------|
| **Búsqueda web** | Información actualizada |
| **Calculadora** | Cálculos matemáticos precisos |
| **Leer archivos** | Analizar documentos que le pasas |
| **Ejecutar código** | Procesar datos, crear gráficos |
| **Enviar emails** | Automatizar comunicaciones |
| **Acceder a APIs** | Conectar con otros servicios |

---

## 5. Tipos de Agentes

### Agente Simple
Un solo agente con varias herramientas.
> "Asistente personal que puede buscar, calcular y tomar notas"

### Agentes en Equipo
Varios agentes especializados que colaboran.
> "Un investigador busca info → Un escritor redacta → Un editor revisa"

### Agente con Memoria
Recuerda conversaciones pasadas y aprende de ellas.
> "Asistente que recuerda tus preferencias"

---

## 6. Limitaciones de los Agentes

| Problema | Por qué pasa |
|----------|--------------|
| **Se equivocan** | El LLM puede elegir mal qué herramienta usar |
| **Se atascan** | Pueden entrar en bucles infinitos |
| **Inventan datos** | Heredan las alucinaciones del LLM |
| **Son lentos** | Cada paso requiere llamar al modelo |
| **Cuestan dinero** | Muchas llamadas al LLM = más coste |

---

## 7. Cuándo Usar un Agente

**SÍ usar cuando:**
- La tarea tiene varios pasos
- Necesitas información externa (web, archivos)
- La tarea varía según el contexto

**NO usar cuando:**
- Una simple pregunta-respuesta basta
- Necesitas 100% precisión
- El coste es importante

---

## 8. Practica

**Ejercicio 1: Prueba un agente con herramientas**
1. Abre ChatGPT (versión gratuita)
2. Pregunta: "¿Cuál es la noticia más importante de hoy en España?"
3. Observa cómo busca en internet antes de responder

**Ejercicio 2: Compara agente vs chatbot**
1. En ChatGPT: "Analiza las ventas de este archivo" (sube un Excel simple)
2. El agente usará herramientas para leerlo y analizarlo

**Ejercicio 3: Agente multi-paso**
> "Busca las 3 películas más taquilleras de 2024, encuentra sus puntuaciones en IMDB, y recomiéndame cuál ver primero basándote en mis gustos: me gustan las películas de acción pero no las muy largas"

---

## 9. Resumen

- **Agente** = IA que ejecuta acciones, no solo habla
- **Herramientas** = habilidades del agente (buscar, calcular, leer archivos)
- **Ciclo**: Pensar → Actuar → Observar → Repetir
- **Ya los usas**: ChatGPT, Claude, Gemini tienen capacidades de agente
- **Limitaciones**: se equivocan, son lentos, pueden ser costosos

