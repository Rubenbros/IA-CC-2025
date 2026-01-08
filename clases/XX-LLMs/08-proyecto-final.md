# Proyecto Final

## 1. Objetivo

Demostrar que entiendes los conceptos del curso creando o analizando una aplicación que use agentes de IA.

---

## 2. Requisitos Mínimos

| Requisito | Descripción |
|-----------|-------------|
| **Usar un framework** | LangChain, CrewAI, o similar |
| **Tener herramientas** | Al menos 2 herramientas/funciones |
| **Interfaz** | Que se pueda usar (web, chat, línea de comandos) |
| **Documentación** | README que explique qué hace y cómo usarlo |

---

## 3. Ideas de Proyectos

### Nivel Básico
- **Asistente de tareas**: crear, listar, completar tareas
- **Chatbot FAQ**: responde sobre un tema usando documentos
- **Calculadora inteligente**: resuelve problemas explicando el proceso

### Nivel Intermedio
- **Generador de contenido**: crea posts para redes sociales
- **Asistente de código**: explica y mejora código
- **Resumidor de noticias**: busca y resume noticias de un tema

### Nivel Avanzado
- **Investigador automático**: investiga un tema y genera un informe
- **Tutor personalizado**: adapta explicaciones al nivel del estudiante
- **Asistente con RAG**: responde usando tus propios documentos

---

## 4. Estructura Sugerida

```
mi-proyecto/
├── README.md           ← Qué hace, cómo instalar, cómo usar
├── requirements.txt    ← pip install -r requirements.txt
├── .env.example        ← GROQ_API_KEY=tu_key_aqui
├── .gitignore          ← .env, __pycache__/
└── app.py              ← Tu código principal
```

---

## 5. Criterios de Evaluación

| Criterio | Peso | Qué se evalúa |
|----------|------|---------------|
| **Funciona** | 30% | El proyecto hace lo que dice |
| **Comprensión** | 30% | Entiendes qué hace y por qué |
| **Código** | 20% | Organizado, sin API keys expuestas |
| **Documentación** | 20% | README claro, instrucciones |

---

## 6. Checklist de Entrega

```
[ ] El proyecto se ejecuta sin errores
[ ] Hay un README con instrucciones
[ ] Las API keys NO están en el código
[ ] requirements.txt incluye todas las dependencias
[ ] .gitignore excluye .env
[ ] Puedo explicar cómo funciona
```

---

## 7. Despliegue Gratuito (Opcional)

Si quieres que otros puedan probarlo:

| Plataforma | Para qué | URL |
|------------|----------|-----|
| **HuggingFace Spaces** | Apps Gradio/Streamlit | huggingface.co/spaces |
| **Streamlit Cloud** | Apps Streamlit | streamlit.io/cloud |

---

## 8. Consejos

- **Empieza simple**: primero que funcione, luego mejora
- **Prueba antes de presentar**: asegúrate de que todo funciona
- **Ten ejemplos listos**: prepara casos que sabes que funcionan
- **Conoce tu código**: prepárate para explicar qué hace cada parte

---

## 9. Recursos

| Recurso | URL |
|---------|-----|
| LangChain Docs | python.langchain.com |
| CrewAI Docs | docs.crewai.com |
| Groq (API gratis) | console.groq.com |
| Gradio (interfaces) | gradio.app |

