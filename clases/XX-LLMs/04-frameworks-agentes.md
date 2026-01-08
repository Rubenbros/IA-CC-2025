# Frameworks para Agentes

## 1. ¿Qué es un Framework?

Un **framework** es una "caja de herramientas" que facilita crear aplicaciones con IA. En lugar de programar todo desde cero, usas componentes ya hechos.

**Analogía:** Es como usar IKEA en vez de fabricar tus propios muebles. Los componentes vienen listos, tú solo los ensamblas.

---

## 2. Los 3 Frameworks Principales

### LangChain
**El más popular y versátil.**
- Sirve para casi todo: chatbots, agentes, RAG
- Muchas integraciones (OpenAI, Google, bases de datos...)
- Gran comunidad y documentación
- Curva de aprendizaje media

**Mejor para:** Proyectos flexibles, cuando necesitas muchas opciones.

### CrewAI
**El más fácil para equipos de agentes.**
- Piensa en "roles" como un equipo real
- Defines agentes como personas: rol, objetivo, personalidad
- Muy intuitivo
- Menos flexible pero más simple

**Mejor para:** Cuando quieres varios agentes trabajando juntos (investigador + escritor + editor).

### AutoGen (Microsoft)
**Para agentes que "conversan" entre ellos.**
- Los agentes se envían mensajes como en un chat
- Bueno para debates o colaboración
- Puede ejecutar código automáticamente
- Respaldado por Microsoft

**Mejor para:** Cuando quieres que agentes "discutan" un problema.

---

## 3. Comparativa Rápida

| Aspecto | LangChain | CrewAI | AutoGen |
|---------|-----------|--------|---------|
| **Facilidad** | Media | Fácil | Media |
| **Flexibilidad** | Muy alta | Media | Alta |
| **Multi-agente** | Posible | Nativo | Nativo |
| **Documentación** | Excelente | Buena | Buena |
| **Comunidad** | Muy grande | Creciendo | Grande |

---

## 4. ¿Cuál Elegir?

**Elige LangChain si:**
- Es tu primer framework (más recursos de aprendizaje)
- Necesitas máxima flexibilidad
- Quieres hacer RAG (búsqueda en documentos)

**Elige CrewAI si:**
- Quieres algo simple y rápido
- Tu problema se modela como "equipo de personas"
- Prefieres menos código

**Elige AutoGen si:**
- Quieres agentes que debatan
- Necesitas ejecutar código automáticamente
- Te gusta el ecosistema Microsoft

---

## 5. Conceptos Clave por Framework

### LangChain
| Concepto | Qué es |
|----------|--------|
| **Chain** | Secuencia de pasos (prompt → LLM → respuesta) |
| **Agent** | LLM que decide qué herramientas usar |
| **Tool** | Función que el agente puede llamar |
| **Memory** | Guarda historial de conversación |

### CrewAI
| Concepto | Qué es |
|----------|--------|
| **Agent** | Un "trabajador" con rol y objetivo |
| **Task** | Una tarea asignada a un agente |
| **Crew** | El equipo completo de agentes |
| **Process** | Cómo trabajan (secuencial o jerárquico) |

### AutoGen
| Concepto | Qué es |
|----------|--------|
| **AssistantAgent** | Agente que responde con IA |
| **UserProxyAgent** | Representa al usuario, puede ejecutar código |
| **GroupChat** | Conversación entre varios agentes |

---

## 6. Otros Frameworks Útiles

| Framework | Para qué |
|-----------|----------|
| **LlamaIndex** | Especializado en RAG (buscar en documentos) |
| **Haystack** | Pipelines de procesamiento de texto |
| **Semantic Kernel** | Para aplicaciones .NET/C# |

---

## 7. Recursos Gratuitos

| Framework | Documentación | Dónde aprender |
|-----------|---------------|----------------|
| **LangChain** | python.langchain.com | Curso en DeepLearning.AI |
| **CrewAI** | docs.crewai.com | Videos en YouTube oficiales |
| **AutoGen** | microsoft.github.io/autogen | Ejemplos en GitHub |

---

## 8. Resumen

- **Framework** = herramientas pre-hechas para crear agentes
- **LangChain**: el más completo y flexible
- **CrewAI**: el más fácil para equipos de agentes
- **AutoGen**: para agentes que conversan entre sí
- **Empieza con LangChain** si no sabes cuál elegir
- Todos son **gratuitos y open source**

