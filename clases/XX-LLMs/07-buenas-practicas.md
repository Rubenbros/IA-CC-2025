# Buenas Prácticas

## 1. Principios Básicos

### Empieza Simple
- Primero haz que funcione lo básico
- Añade complejidad poco a poco
- No intentes hacer todo a la vez

### Prueba Frecuentemente
- Prueba cada cambio antes de seguir
- No esperes a tener todo listo para probar
- Si algo falla, es más fácil saber qué fue

### Documenta mientras Haces
- Apunta qué hace cada parte
- Tu yo del futuro te lo agradecerá
- No confíes en tu memoria

---

## 2. Seguridad: API Keys

**NUNCA** pongas tus API keys directamente en el código.

| ❌ Mal | ✅ Bien |
|--------|---------|
| `api_key = "sk-abc123..."` | Usar variables de entorno |
| Subir .env a GitHub | Añadir .env a .gitignore |
| Compartir keys | Cada persona su propia key |

**Cómo hacerlo bien:**
1. Crea un archivo `.env` con tus keys
2. Añade `.env` a `.gitignore`
3. Usa una librería para leer las variables

---

## 3. Manejo de Errores

Las APIs de IA pueden fallar. Prepárate para:

| Error | Qué hacer |
|-------|-----------|
| **Rate limit** | Esperar y reintentar |
| **Timeout** | Poner un límite de tiempo |
| **API caída** | Tener un plan B (otro servicio) |
| **Respuesta rara** | Validar antes de usar |

---

## 4. Optimización de Costes

**Para APIs de pago:**
- Usa modelos pequeños para tareas simples
- Guarda respuestas frecuentes (caché)
- Limita la longitud de respuestas
- No envíes contexto innecesario

**Servicios gratuitos:**
| Servicio | Qué ofrece gratis |
|----------|------------------|
| Groq | Muy generoso |
| Google AI Studio | 60 peticiones/minuto |
| Ollama | Todo (corre en tu PC) |

---

## 5. Estructura de un Proyecto

```
mi-proyecto/
├── README.md          ← Qué es y cómo usarlo
├── requirements.txt   ← Dependencias
├── .env.example       ← Template de variables (SIN valores reales)
├── .gitignore         ← Qué NO subir a git
├── app.py             ← Código principal
└── src/               ← Código adicional
```

### .gitignore mínimo
```
.env
*.pyc
__pycache__/
```

---

## 6. Checklist antes de Entregar

```
CÓDIGO:
[ ] ¿Funciona sin errores?
[ ] ¿Las API keys están en .env?
[ ] ¿Hay un requirements.txt?

DOCUMENTACIÓN:
[ ] ¿El README explica qué hace?
[ ] ¿Hay instrucciones de instalación?
[ ] ¿Se explica cómo configurar las keys?

FUNCIONAMIENTO:
[ ] ¿Probé los casos principales?
[ ] ¿Qué pasa si algo falla?
```

---

## 7. Resumen

- **Empieza simple**, complica después
- **API keys en .env**, nunca en el código
- **Prueba frecuentemente**
- **Documenta** lo que haces
- **Prepárate para errores** de la API
- **Usa servicios gratuitos**: Groq, Google AI, Ollama

