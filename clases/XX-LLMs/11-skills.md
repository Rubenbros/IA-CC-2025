# Skills (Comandos Especializados)

## 1. ¿Qué son los Skills?

Los **Skills** son comandos predefinidos que activan comportamientos especializados en el agente de IA. Se invocan con el prefijo `/` (slash commands).

**Analogía:** Son como atajos de teclado, pero para tareas complejas. En lugar de explicar paso a paso lo que quieres, ejecutas un comando y el agente sabe exactamente qué hacer.

```
Sin Skill:
"Revisa los cambios que he hecho, crea un mensaje de commit
descriptivo siguiendo conventional commits, y haz el commit"

Con Skill:
/commit
```

---

## 2. ¿Por qué usar Skills?

| Ventaja | Explicación |
|---------|-------------|
| **Rapidez** | Un comando en lugar de escribir instrucciones largas |
| **Consistencia** | Siempre hace la tarea de la misma forma |
| **Menos errores** | Procedimiento probado y optimizado |
| **Flujo de trabajo** | Se integra naturalmente en tu proceso |

---

## 3. Skills Disponibles

### 3.1 Skills de Git

| Skill | Qué hace |
|-------|----------|
| `/commit` | Analiza cambios y crea un commit con mensaje descriptivo |
| `/pr` | Crea un Pull Request con descripción automática |
| `/review-pr` | Revisa un PR y da feedback |

### 3.2 Skills de Código

| Skill | Qué hace |
|-------|----------|
| `/fix` | Busca y corrige errores en el código |
| `/refactor` | Mejora código sin cambiar funcionalidad |
| `/test` | Genera tests para el código |
| `/docs` | Genera documentación |

### 3.3 Skills de Proyecto

| Skill | Qué hace |
|-------|----------|
| `/init` | Inicializa un nuevo proyecto |
| `/setup` | Configura el entorno de desarrollo |
| `/help` | Muestra ayuda y comandos disponibles |

---

## 4. Instalación en Claude Code

### 4.1 Skills incluidos por defecto

Claude Code incluye skills básicos sin configuración:

```bash
# Verificar Claude Code
claude --version

# Ver skills disponibles
claude /help
```

### 4.2 Usar un Skill

Simplemente escribe el comando:

```bash
# En la conversación con Claude Code
/commit

# Con argumentos
/pr 123
/review-pr https://github.com/user/repo/pull/456
```

### 4.3 Configurar Skills personalizados

Crea el archivo de configuración:

| Sistema | Ruta |
|---------|------|
| **Windows** | `%USERPROFILE%\.claude\skills.json` |
| **Mac/Linux** | `~/.claude/skills.json` |

**Ejemplo de skill personalizado:**

```json
{
  "skills": {
    "deploy": {
      "description": "Despliega la aplicación a producción",
      "steps": [
        "Ejecuta los tests",
        "Construye la aplicación",
        "Sube a producción",
        "Verifica que funciona"
      ]
    },
    "morning": {
      "description": "Rutina de inicio del día",
      "steps": [
        "git pull origin main",
        "npm install",
        "Muestra PRs pendientes de revisar",
        "Lista tareas del día"
      ]
    }
  }
}
```

---

## 5. Instalación en OpenCode

### 5.1 Verificar instalación

```bash
# Ver versión
opencode --version

# Ver skills disponibles
opencode /help
```

### 5.2 Configurar Skills

Archivo de configuración:

| Sistema | Ruta |
|---------|------|
| **Windows** | `%USERPROFILE%\.opencode\skills.json` |
| **Mac/Linux** | `~/.opencode/skills.json` |

**Ejemplo de configuración:**

```json
{
  "skills": {
    "lint": {
      "description": "Ejecuta linter y corrige errores",
      "command": "npm run lint:fix"
    },
    "update-deps": {
      "description": "Actualiza dependencias de forma segura",
      "steps": [
        "npm outdated",
        "Analiza qué actualizar",
        "npm update",
        "Ejecuta tests para verificar"
      ]
    }
  }
}
```

---

## 6. Cómo Usar Skills

### 6.1 Uso básico

```
/commit                    → Crea commit de los cambios actuales
/pr                        → Crea Pull Request
/help                      → Muestra ayuda
```

### 6.2 Con argumentos

```
/review-pr 123             → Revisa el PR #123
/fix "error de login"      → Busca y corrige ese error específico
/test auth.js              → Genera tests para ese archivo
```

### 6.3 Encadenar skills

```
/lint && /test && /commit  → Lint, test y commit en secuencia
```

---

## 7. Skill: /commit (Detalle)

El skill más usado. Analiza tus cambios y crea un commit profesional.

### Qué hace internamente:

```
1. git status              → Ve qué archivos cambiaron
2. git diff                → Analiza los cambios
3. Genera mensaje          → Siguiendo conventional commits
4. git add + git commit    → Ejecuta el commit
```

### Ejemplo de uso:

```
Tú: /commit

Claude Code:
- Analizando cambios...
- Archivos modificados: auth.js, login.test.js
- Cambio detectado: corrección de bug en validación

Commit creado:
"fix(auth): corregir validación de token expirado

- Añadida comprobación de fecha de expiración
- Añadido test para tokens expirados"
```

### Formato de mensajes (Conventional Commits):

| Tipo | Cuándo usar |
|------|-------------|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `docs` | Cambios en documentación |
| `style` | Formato, no afecta lógica |
| `refactor` | Mejora sin cambiar funcionalidad |
| `test` | Añadir o modificar tests |
| `chore` | Mantenimiento, dependencias |

---

## 8. Skill: /pr (Detalle)

Crea un Pull Request con descripción automática.

### Qué hace internamente:

```
1. Analiza commits         → Ve qué cambios incluye el PR
2. git diff main...HEAD    → Compara con rama principal
3. Genera descripción      → Resumen, cambios, testing
4. gh pr create            → Crea el PR en GitHub
```

### Ejemplo de uso:

```
Tú: /pr

Claude Code:
- Rama actual: feature/dark-mode
- 3 commits desde main
- Creando PR...

PR creado: #45 "Añadir modo oscuro"
https://github.com/user/repo/pull/45
```

### Estructura del PR generado:

```markdown
## Resumen
Implementa modo oscuro en la aplicación

## Cambios
- Añadido toggle en settings
- Creado hook useTheme
- Estilos CSS para dark mode

## Testing
- [ ] Verificar toggle funciona
- [ ] Comprobar persistencia en localStorage
- [ ] Revisar contraste de colores
```

---

## 9. Skill: /review-pr (Detalle)

Revisa un Pull Request y da feedback.

### Qué hace internamente:

```
1. Obtiene el PR           → Lee código y descripción
2. Analiza cambios         → Busca problemas potenciales
3. Genera review           → Comentarios constructivos
```

### Ejemplo de uso:

```
Tú: /review-pr 123

Claude Code:
Revisando PR #123 "Añadir autenticación OAuth"...

📋 Resumen:
- 5 archivos modificados
- +250 / -30 líneas

✅ Puntos positivos:
- Buena separación de responsabilidades
- Tests incluidos

⚠️ Sugerencias:
- Línea 45 auth.js: Token debería tener expiración
- Falta manejo de error cuando OAuth falla

🔒 Seguridad:
- Verificar que redirect_uri está validado
```

---

## 10. Crear Skills Personalizados

### 10.1 Estructura básica

```json
{
  "skills": {
    "nombre-del-skill": {
      "description": "Qué hace este skill",
      "steps": [
        "Paso 1 que debe hacer",
        "Paso 2 que debe hacer"
      ]
    }
  }
}
```

### 10.2 Ejemplos útiles

**Skill para despliegue:**

```json
{
  "deploy-staging": {
    "description": "Despliega a staging",
    "steps": [
      "Ejecuta npm test",
      "Si pasan, ejecuta npm run build",
      "Despliega con: npm run deploy:staging",
      "Verifica que la app funciona en staging"
    ]
  }
}
```

**Skill para revisión de código:**

```json
{
  "code-review": {
    "description": "Revisa calidad del código",
    "steps": [
      "Ejecuta el linter",
      "Busca código duplicado",
      "Verifica que hay tests",
      "Comprueba que no hay console.log",
      "Genera reporte de mejoras"
    ]
  }
}
```

**Skill para inicio del día:**

```json
{
  "standup": {
    "description": "Prepara el daily standup",
    "steps": [
      "git log --oneline --since='yesterday'",
      "Muestra mis commits de ayer",
      "Lista PRs donde soy reviewer",
      "Muestra issues asignados a mí"
    ]
  }
}
```

---

## 11. Skills vs SubAgentes

| Característica | Skills | SubAgentes |
|----------------|--------|------------|
| **Activación** | Explícita (/comando) | Automática o explícita |
| **Propósito** | Tarea específica predefinida | Tarea dinámica |
| **Personalización** | Defines los pasos | El agente decide los pasos |
| **Ejemplo** | /commit siempre hace commit | Explore busca lo que le pidas |

**Cuándo usar cada uno:**

- **Skills**: Tareas repetitivas que siempre son iguales
- **SubAgentes**: Tareas que varían según el contexto

---

## 12. Skills en el Flujo de Trabajo

### Flujo típico de desarrollo:

```
1. /morning              → Actualizar repo, ver tareas
2. [Escribir código]
3. /lint                 → Verificar estilo
4. /test                 → Ejecutar tests
5. /commit               → Guardar cambios
6. /pr                   → Crear Pull Request
7. /review-pr 45         → Revisar PR de un compañero
```

### Automatizar con aliases:

```bash
# En tu .bashrc o .zshrc
alias cc="claude"
alias commit="claude /commit"
alias pr="claude /pr"
```

---

## 13. Troubleshooting

### El skill no se reconoce

```bash
# Verificar skills disponibles
/help

# Verificar archivo de configuración
cat ~/.claude/skills.json
```

### El skill falla

```
# Ejecutar con más detalle
/commit --verbose

# Ver logs
claude --debug /commit
```

### Crear skill que no funciona

Verifica que el JSON es válido:
```bash
# Validar JSON
cat ~/.claude/skills.json | python -m json.tool
```

---

## 14. Resumen

- **Skills** = Comandos predefinidos que activan tareas específicas
- **Invocación**: Con `/` (ej: `/commit`, `/pr`, `/help`)
- **Incluidos**: `/commit`, `/pr`, `/review-pr`, `/help` y más
- **Personalizables**: Crea tus propios skills en `skills.json`
- **Ventajas**: Rapidez, consistencia, menos errores
- **Diferencia con SubAgentes**: Skills son tareas fijas, subagentes son dinámicos
- **Flujo de trabajo**: Integra skills en tu rutina diaria de desarrollo
