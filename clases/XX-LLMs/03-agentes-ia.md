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

## 3.1. Arquitecturas de Agentes

### ReAct (Reasoning + Acting)

El patrón más común. El agente alterna entre **razonar** y **actuar**.

```
THOUGHT: ¿Qué necesito hacer?
ACTION: Ejecuto una herramienta
OBSERVATION: Veo el resultado
THOUGHT: ¿Tengo suficiente info?
... (repite hasta terminar)
```

**Ventaja:** Simple y efectivo para la mayoría de tareas.

### Plan-and-Execute (Planificar y Ejecutar)

Primero crea un plan completo, luego lo ejecuta paso a paso.

```
TAREA: "Investiga las 3 mejores laptops para programación y crea una comparativa"

PLAN:
1. Buscar laptops recomendadas para programación
2. Extraer specs de las 3 mejores
3. Comparar características
4. Crear tabla comparativa

EJECUCIÓN: Sigue el plan paso a paso, ajustando si es necesario
```

**Ventaja:** Mejor para tareas complejas con muchos pasos.

### Reflexión y Auto-corrección

El agente revisa su propio trabajo y lo mejora.

```
TAREA: Escribir código para ordenar una lista
PRIMERA VERSIÓN: [código con bug]
REFLEXIÓN: "Este código falla con listas vacías"
VERSIÓN CORREGIDA: [código mejorado]
VERIFICACIÓN: Pruebo con varios casos
```

**Implementación práctica:**
> "Después de completar la tarea, revisa tu trabajo y corrige cualquier error que encuentres."

---

## 3.2. Memoria en Agentes

Los agentes pueden tener diferentes tipos de memoria:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Memoria de trabajo** | La conversación actual | Lo que hablamos ahora |
| **Memoria corto plazo** | Información de la sesión | Tareas anteriores de hoy |
| **Memoria largo plazo** | Información persistente | Tu nombre, preferencias |
| **Memoria episódica** | Experiencias pasadas | "La última vez que hicimos esto..." |

### Implementación Común

**Vectores/Embeddings:** El agente convierte texto en vectores numéricos y los guarda en una base de datos. Cuando necesita recordar algo, busca vectores similares.

```
Usuario: "Me llamo Juan y me gusta Python"
→ Se guarda en memoria a largo plazo

[Días después]
Usuario: "¿Cuál era mi lenguaje favorito?"
→ Busca en memoria → Encuentra "Python"
→ "Tu lenguaje favorito es Python, Juan"
```

**Productos que lo usan:**
- **ChatGPT Memory** - Recuerda cosas entre conversaciones
- **Claude Projects** - Contexto persistente por proyecto
- **Personal AI assistants** - Aprenden de ti con el tiempo

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

### Agente Simple (Single Agent)
Un solo agente con varias herramientas.
> "Asistente personal que puede buscar, calcular y tomar notas"

**Cuándo usarlo:** Tareas sencillas donde un agente puede manejar todo.

### Agentes en Equipo (Multi-Agent Systems)
Varios agentes especializados que colaboran.
> "Un investigador busca info → Un escritor redacta → Un editor revisa"

**Patrones comunes:**

| Patrón | Cómo funciona |
|--------|---------------|
| **Supervisor** | Un agente jefe delega tareas a agentes trabajadores |
| **Pipeline** | Agentes en cadena, cada uno procesa y pasa al siguiente |
| **Debate** | Múltiples agentes discuten hasta llegar a consenso |
| **Especialistas** | Cada agente domina un área (código, datos, texto) |

```
Ejemplo: Sistema de análisis de documentos

[Usuario sube PDF]
    ↓
[Agente Extractor] → Extrae texto y estructura
    ↓
[Agente Analizador] → Identifica temas clave
    ↓
[Agente Resumidor] → Crea resumen ejecutivo
    ↓
[Resultado final al usuario]
```

### Agente con Memoria
Recuerda conversaciones pasadas y aprende de ellas.
> "Asistente que recuerda tus preferencias"

---

## 5.1. Function Calling / Tool Use

**Function Calling** es cómo los LLMs modernos usan herramientas de forma estructurada.

### Cómo funciona

1. Defines las herramientas disponibles (nombre, descripción, parámetros)
2. El modelo decide si necesita usar una
3. Genera una llamada estructurada (JSON)
4. Tu código ejecuta la función
5. El resultado vuelve al modelo

```python
# Ejemplo simplificado de definición de herramienta
herramientas = [
    {
        "nombre": "buscar_clima",
        "descripcion": "Busca el clima de una ciudad",
        "parametros": {
            "ciudad": "string (requerido)"
        }
    }
]

# El modelo genera:
{
    "herramienta": "buscar_clima",
    "parametros": {"ciudad": "Madrid"}
}

# Tu código ejecuta y devuelve el resultado
```

**Esto es lo que hace que ChatGPT pueda:**
- Buscar en internet
- Ejecutar código Python
- Generar imágenes con DALL-E
- Leer archivos que subes

---

## 5.2. Agentes de Código (Coding Agents)

Una categoría especial de agentes que pueden **escribir y ejecutar código**.

### Ejemplos Populares

| Herramienta | Descripción |
|-------------|-------------|
| **GitHub Copilot** | Autocompleta código en tu IDE |
| **Cursor** | IDE completo con IA integrada |
| **Claude Code** | Agente que puede modificar archivos y ejecutar comandos |
| **Devin** | "Ingeniero de software IA" (muy avanzado) |
| **Replit Agent** | Crea aplicaciones completas desde cero |

### Capacidades Típicas
- Leer y entender código existente
- Escribir código nuevo
- Ejecutar tests
- Debuggear errores
- Refactorizar
- Crear proyectos completos

**El futuro:** Los coding agents están evolucionando de "autocompletado" a "compañero de desarrollo" que puede hacer tareas completas

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

## 8. Agentes en el Mundo Real (2024-2025)

### Casos de Uso Actuales

| Sector | Aplicación |
|--------|------------|
| **Desarrollo** | Coding assistants, code review automático, debugging |
| **Atención al cliente** | Chatbots que resuelven problemas complejos |
| **Investigación** | Análisis de papers, síntesis de información |
| **Productividad** | Automatización de emails, scheduling, tareas repetitivas |
| **Datos** | Análisis, visualización, reportes automáticos |

### El Futuro Cercano

- **Computer Use**: Agentes que controlan tu ordenador (hacen clic, escriben, navegan)
- **Agentes autónomos**: Pueden trabajar horas sin supervisión
- **Agentes especializados**: Expertos en dominios específicos (legal, médico, financiero)
- **Equipos de agentes**: Múltiples IAs colaborando en proyectos complejos

**Lo que está cambiando:**
> De "pregunta → respuesta" a "objetivo → el agente hace todo el trabajo"

---

## 9. Resumen

- **Agente** = IA que ejecuta acciones, no solo habla
- **Herramientas** = habilidades del agente (buscar, calcular, leer archivos)
- **Ciclo**: Pensar → Actuar → Observar → Repetir
- **Arquitecturas**: ReAct, Plan-and-Execute, Multi-agente
- **Memoria**: Permite que el agente recuerde y aprenda
- **Function Calling**: Cómo los LLMs usan herramientas de forma estructurada
- **Coding Agents**: Agentes especializados en escribir y modificar código
- **Ya los usas**: ChatGPT, Claude, Gemini tienen capacidades de agente
- **Limitaciones**: se equivocan, son lentos, pueden ser costosos
- **El futuro**: Agentes más autónomos que completan tareas complejas

