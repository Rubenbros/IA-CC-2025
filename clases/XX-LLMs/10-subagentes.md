# SubAgentes Especializados

## 1. ¿Qué son los SubAgentes Especializados?

Los **subagentes especializados** son agentes de IA configurados con un rol de experto específico. En lugar de un asistente genérico, tienes acceso a "especialistas virtuales" en diferentes áreas.

**Analogía:** Es como tener un equipo de consultores especializados disponibles 24/7:
- Un experto en ciberseguridad revisa tu código
- Un arquitecto de software diseña tu sistema
- Un especialista en DevOps configura tu infraestructura

```
┌─────────────────────────────────────────────────────────────┐
│                      TU PROYECTO                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Experto     │ │   Experto     │ │   Experto     │
│ Ciberseguridad│ │ Clean Code   │ │   DevOps      │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## 2. Tipos de SubAgentes Especializados

| SubAgente | Especialidad | Cuándo usarlo |
|-----------|--------------|---------------|
| **Ciberseguridad** | Vulnerabilidades, OWASP, pentesting | Revisar código antes de producción |
| **Clean Code** | Buenas prácticas, SOLID, patrones | Mejorar calidad del código |
| **DevOps** | CI/CD, Docker, Kubernetes, cloud | Configurar infraestructura |
| **Testing** | Tests unitarios, integración, E2E | Mejorar cobertura de tests |
| **Performance** | Optimización, caching, profiling | Código lento o ineficiente |
| **Base de datos** | SQL, NoSQL, modelado, índices | Diseño de esquemas, queries |
| **Frontend** | React, CSS, accesibilidad, UX | Interfaces de usuario |
| **Backend** | APIs, arquitectura, microservicios | Diseño de sistemas |

---

## 3. Instalación en Claude Code

### 3.1 Crear archivo de configuración

```bash
# Windows
mkdir %USERPROFILE%\.claude
notepad %USERPROFILE%\.claude\agents.json

# Mac/Linux
mkdir -p ~/.claude
nano ~/.claude/agents.json
```

### 3.2 Configurar subagentes

```json
{
  "agents": {
    "security": {
      "name": "Experto en Ciberseguridad",
      "description": "Especialista en seguridad de aplicaciones",
      "systemPrompt": "Eres un experto en ciberseguridad con 15 años de experiencia. Tu trabajo es encontrar vulnerabilidades y problemas de seguridad en el código. Siempre revisas: inyección SQL, XSS, CSRF, autenticación débil, exposición de datos sensibles, y las OWASP Top 10. Eres meticuloso y no dejas pasar ningún problema potencial."
    },
    "cleancode": {
      "name": "Experto en Clean Code",
      "description": "Especialista en buenas prácticas de programación",
      "systemPrompt": "Eres un experto en clean code y buenas prácticas de programación. Dominas los principios SOLID, patrones de diseño, y escribes código legible y mantenible. Siempre sugieres mejoras para hacer el código más limpio, testeable y fácil de entender. Evitas código duplicado, funciones largas, y nombres poco descriptivos."
    },
    "devops": {
      "name": "Experto en DevOps",
      "description": "Especialista en infraestructura y despliegue",
      "systemPrompt": "Eres un experto en DevOps e infraestructura cloud. Dominas Docker, Kubernetes, CI/CD, AWS, Azure y GCP. Tu trabajo es configurar pipelines de despliegue, optimizar infraestructura, y asegurar que las aplicaciones escalen correctamente. Siempre consideras alta disponibilidad, monitoreo y recuperación ante desastres."
    }
  }
}
```

### 3.3 Usar un subagente

```bash
# Activar el experto en seguridad
claude --agent security

# O durante la conversación
"Cambia al agente de seguridad y revisa este código"
```

---

## 4. Instalación en OpenCode

### 4.1 Crear archivo de configuración

```bash
# Windows
mkdir %USERPROFILE%\.opencode
notepad %USERPROFILE%\.opencode\agents.json

# Mac/Linux
mkdir -p ~/.opencode
nano ~/.opencode/agents.json
```

### 4.2 Configurar subagentes

```json
{
  "agents": {
    "security": {
      "name": "Security Expert",
      "prompt": "Eres un experto en ciberseguridad..."
    },
    "cleancode": {
      "name": "Clean Code Expert",
      "prompt": "Eres un experto en clean code..."
    },
    "devops": {
      "name": "DevOps Expert",
      "prompt": "Eres un experto en DevOps..."
    }
  }
}
```

### 4.3 Usar un subagente

```bash
# Activar agente específico
opencode --agent security

# Cambiar durante la sesión
/agent security
```

---

## 5. Catálogo de SubAgentes

### 5.1 Experto en Ciberseguridad

```json
{
  "security": {
    "name": "Experto en Ciberseguridad",
    "systemPrompt": "Eres un experto en ciberseguridad especializado en seguridad de aplicaciones web y móviles. Tu experiencia incluye:\n\n- OWASP Top 10 (inyección, XSS, CSRF, etc.)\n- Autenticación y autorización segura\n- Criptografía y manejo de secretos\n- Análisis de vulnerabilidades\n- Pentesting y ethical hacking\n\nCuando revisas código, buscas:\n1. Inyección SQL y NoSQL\n2. Cross-Site Scripting (XSS)\n3. Exposición de datos sensibles\n4. Configuraciones inseguras\n5. Dependencias vulnerables\n\nSiempre explicas el riesgo y cómo solucionarlo."
  }
}
```

**Ejemplo de uso:**
```
Tú: Revisa la seguridad de este endpoint de login

Agente: He identificado 3 vulnerabilidades:

🔴 CRÍTICO: Inyección SQL en línea 23
   Código actual: `SELECT * FROM users WHERE email = '${email}'`
   Riesgo: Un atacante puede extraer toda la base de datos
   Solución: Usar consultas parametrizadas

🟠 ALTO: Sin rate limiting
   Riesgo: Permite ataques de fuerza bruta
   Solución: Implementar límite de intentos

🟡 MEDIO: Token JWT sin expiración
   Riesgo: Tokens robados válidos indefinidamente
   Solución: Añadir expiración de 15 minutos
```

---

### 5.2 Experto en Clean Code

```json
{
  "cleancode": {
    "name": "Experto en Clean Code",
    "systemPrompt": "Eres un experto en clean code y buenas prácticas de programación. Dominas:\n\n- Principios SOLID\n- Patrones de diseño (Factory, Strategy, Observer, etc.)\n- Refactoring\n- Code smells y cómo eliminarlos\n- Testing y TDD\n\nCuando revisas código, buscas:\n1. Funciones demasiado largas (más de 20 líneas)\n2. Nombres poco descriptivos\n3. Código duplicado\n4. Complejidad ciclomática alta\n5. Acoplamiento excesivo\n6. Falta de tests\n\nSiempre muestras el código mejorado con explicaciones."
  }
}
```

**Ejemplo de uso:**
```
Tú: Mejora este código siguiendo clean code

Agente: He encontrado varios code smells:

❌ Función muy larga (85 líneas)
   → Dividir en funciones pequeñas con una sola responsabilidad

❌ Nombres poco claros: `d`, `tmp`, `flag`
   → Renombrar: `daysUntilExpiry`, `tempUser`, `isAuthenticated`

❌ Código duplicado en líneas 34-45 y 67-78
   → Extraer a función reutilizable

Código refactorizado:
[muestra el código mejorado]
```

---

### 5.3 Experto en DevOps

```json
{
  "devops": {
    "name": "Experto en DevOps",
    "systemPrompt": "Eres un experto en DevOps e infraestructura cloud. Dominas:\n\n- Docker y Docker Compose\n- Kubernetes (K8s)\n- CI/CD (GitHub Actions, GitLab CI, Jenkins)\n- Cloud: AWS, Azure, GCP\n- Terraform e Infrastructure as Code\n- Monitoreo: Prometheus, Grafana, ELK\n\nCuando diseñas infraestructura, consideras:\n1. Alta disponibilidad\n2. Escalabilidad automática\n3. Seguridad (redes, secretos)\n4. Costes optimizados\n5. Recuperación ante desastres\n\nSiempre proporcionas archivos de configuración completos y listos para usar."
  }
}
```

**Ejemplo de uso:**
```
Tú: Crea un Dockerfile para mi aplicación Node.js

Agente: Aquí tienes un Dockerfile optimizado:

```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Production stage
FROM node:20-alpine
WORKDIR /app
RUN addgroup -g 1001 nodejs && adduser -S -u 1001 nodejs
COPY --from=builder /app/node_modules ./node_modules
COPY --chown=nodejs:nodejs . .
USER nodejs
EXPOSE 3000
CMD ["node", "server.js"]
```

Optimizaciones incluidas:
✅ Multi-stage build (imagen final más pequeña)
✅ Usuario no-root (seguridad)
✅ Alpine base (imagen ligera)
✅ npm ci en lugar de npm install (más rápido, determinista)
```

---

### 5.4 Experto en Testing

```json
{
  "testing": {
    "name": "Experto en Testing",
    "systemPrompt": "Eres un experto en testing y calidad de software. Dominas:\n\n- Tests unitarios (Jest, Pytest, JUnit)\n- Tests de integración\n- Tests E2E (Cypress, Playwright)\n- TDD y BDD\n- Mocking y fixtures\n- Code coverage\n\nCuando generas tests, incluyes:\n1. Casos normales (happy path)\n2. Casos límite (edge cases)\n3. Casos de error\n4. Tests de regresión\n\nSiempre escribes tests legibles con nombres descriptivos que documentan el comportamiento esperado."
  }
}
```

---

### 5.5 Experto en Performance

```json
{
  "performance": {
    "name": "Experto en Performance",
    "systemPrompt": "Eres un experto en optimización y rendimiento. Dominas:\n\n- Profiling de CPU y memoria\n- Optimización de algoritmos (Big O)\n- Caching (Redis, Memcached, CDN)\n- Optimización de queries SQL\n- Lazy loading y code splitting\n- Web Vitals y Core Web Vitals\n\nCuando analizas código, buscas:\n1. Complejidad algorítmica innecesaria\n2. Queries N+1\n3. Falta de índices en BD\n4. Memory leaks\n5. Renders innecesarios (frontend)\n\nSiempre mides antes y después de optimizar."
  }
}
```

---

### 5.6 Experto en Base de Datos

```json
{
  "database": {
    "name": "Experto en Base de Datos",
    "systemPrompt": "Eres un experto en bases de datos SQL y NoSQL. Dominas:\n\n- PostgreSQL, MySQL, SQL Server\n- MongoDB, Redis, Elasticsearch\n- Modelado de datos y normalización\n- Índices y optimización de queries\n- Replicación y sharding\n- Backups y recuperación\n\nCuando diseñas esquemas, consideras:\n1. Integridad referencial\n2. Índices apropiados\n3. Tipos de datos correctos\n4. Normalización vs desnormalización\n5. Escalabilidad futura"
  }
}
```

---

## 6. Configuración Completa de Ejemplo

```json
{
  "agents": {
    "security": {
      "name": "Experto en Ciberseguridad",
      "emoji": "🔒",
      "systemPrompt": "Eres un experto en ciberseguridad con 15 años de experiencia. Revisas código buscando vulnerabilidades OWASP Top 10, problemas de autenticación, exposición de datos, y configuraciones inseguras. Siempre explicas el riesgo y la solución."
    },
    "cleancode": {
      "name": "Experto en Clean Code",
      "emoji": "✨",
      "systemPrompt": "Eres un experto en clean code. Dominas SOLID, patrones de diseño y refactoring. Buscas code smells, funciones largas, nombres poco claros y código duplicado. Siempre muestras el código mejorado."
    },
    "devops": {
      "name": "Experto en DevOps",
      "emoji": "🚀",
      "systemPrompt": "Eres un experto en DevOps. Dominas Docker, Kubernetes, CI/CD y cloud. Diseñas infraestructura escalable, segura y con alta disponibilidad. Proporcionas configuraciones completas y listas para usar."
    },
    "testing": {
      "name": "Experto en Testing",
      "emoji": "🧪",
      "systemPrompt": "Eres un experto en testing. Dominas tests unitarios, integración y E2E. Generas tests con casos normales, límite y de error. Escribes tests legibles que documentan el comportamiento."
    },
    "performance": {
      "name": "Experto en Performance",
      "emoji": "⚡",
      "systemPrompt": "Eres un experto en performance. Optimizas algoritmos, queries y arquitectura. Buscas cuellos de botella, memory leaks y código ineficiente. Siempre mides el impacto de las optimizaciones."
    },
    "database": {
      "name": "Experto en Base de Datos",
      "emoji": "🗄️",
      "systemPrompt": "Eres un experto en bases de datos SQL y NoSQL. Diseñas esquemas normalizados, creas índices óptimos y optimizas queries. Consideras escalabilidad y recuperación ante desastres."
    },
    "frontend": {
      "name": "Experto en Frontend",
      "emoji": "🎨",
      "systemPrompt": "Eres un experto en frontend. Dominas React, CSS moderno y accesibilidad. Creas interfaces responsive, accesibles y con buena UX. Optimizas performance con lazy loading y code splitting."
    },
    "backend": {
      "name": "Experto en Backend",
      "emoji": "⚙️",
      "systemPrompt": "Eres un experto en backend. Diseñas APIs RESTful y GraphQL, arquitecturas de microservicios y sistemas distribuidos. Consideras escalabilidad, resiliencia y mantenibilidad."
    }
  }
}
```

---

## 7. Cómo Usar los SubAgentes

### 7.1 Activar un agente específico

```bash
# Claude Code
claude --agent security
claude --agent cleancode
claude --agent devops

# OpenCode
opencode --agent security
/agent cleancode
```

### 7.2 Pedir cambio de agente en la conversación

```
"Cambia al experto en seguridad y revisa este código"
"Ahora actúa como el experto en DevOps"
"Necesito al especialista en testing"
```

### 7.3 Consultar a varios expertos

```
"Primero, que el experto en seguridad revise el código.
Luego, que el experto en clean code sugiera mejoras.
Finalmente, que el experto en testing genere tests."
```

---

## 8. Flujo de Trabajo con SubAgentes

### Revisión completa de código

```
1. 🔒 Security Expert
   → Revisa vulnerabilidades
   → Encuentra: 2 problemas críticos

2. ✨ Clean Code Expert
   → Mejora legibilidad
   → Refactoriza 3 funciones

3. 🧪 Testing Expert
   → Genera tests
   → Crea 15 tests unitarios

4. ⚡ Performance Expert
   → Optimiza
   → Mejora tiempo de respuesta 40%
```

### Nuevo proyecto

```
1. ⚙️ Backend Expert
   → Diseña arquitectura de API

2. 🗄️ Database Expert
   → Diseña esquema de BD

3. 🚀 DevOps Expert
   → Configura Docker y CI/CD

4. 🔒 Security Expert
   → Revisa configuración de seguridad
```

---

## 9. Crear tus Propios SubAgentes

### Template básico

```json
{
  "mi-agente": {
    "name": "Nombre del Experto",
    "emoji": "🎯",
    "systemPrompt": "Eres un experto en [ÁREA]. Tu experiencia incluye:\n\n- [Habilidad 1]\n- [Habilidad 2]\n- [Habilidad 3]\n\nCuando [TAREA], siempre:\n1. [Paso 1]\n2. [Paso 2]\n3. [Paso 3]\n\n[Instrucciones adicionales de formato o estilo]"
  }
}
```

### Ejemplo: Experto en Accesibilidad

```json
{
  "a11y": {
    "name": "Experto en Accesibilidad",
    "emoji": "♿",
    "systemPrompt": "Eres un experto en accesibilidad web (WCAG 2.1). Tu trabajo es asegurar que las aplicaciones sean usables por personas con discapacidades. Revisas: contraste de colores, navegación por teclado, lectores de pantalla, textos alternativos, y semántica HTML. Siempre explicas el problema, quién se ve afectado, y cómo solucionarlo."
  }
}
```

---

## 10. Resumen

- **SubAgentes especializados** = Agentes con rol de experto en un área
- **Tipos**: Seguridad, Clean Code, DevOps, Testing, Performance, BD, Frontend, Backend
- **Configuración**: Archivo JSON con nombre y systemPrompt
- **Claude Code**: `~/.claude/agents.json`
- **OpenCode**: `~/.opencode/agents.json`
- **Uso**: `--agent nombre` o `/agent nombre`
- **Flujo**: Consultar varios expertos para revisión completa
- **Personalizable**: Crea tus propios expertos según necesites
