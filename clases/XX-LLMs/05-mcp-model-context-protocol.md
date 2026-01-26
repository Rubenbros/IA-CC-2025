# MCP (Model Context Protocol)

## 1. ¿Qué es MCP?

**MCP** es un estándar abierto creado por Anthropic (los creadores de Claude) para conectar IAs con herramientas externas de forma universal.

**Problema que resuelve:**
Antes, cada aplicación tenía que programar sus propias conexiones con herramientas. Con MCP, se hace una vez y funciona para todas.

**Analogía:** MCP es como el USB. Antes cada dispositivo tenía su propio conector, ahora con USB todo se conecta igual.

```
┌─────────────────┐     MCP Protocol      ┌─────────────────┐
│   Claude/LLM    │◄────────────────────► │   MCP Server    │
│    (Cliente)    │    (JSON-RPC 2.0)     │  (Herramienta)  │
└─────────────────┘                        └─────────────────┘
         │                                         │
         │                                         ▼
         │                                 ┌───────────────┐
         │                                 │  Sistema Real │
         │                                 │  - Archivos   │
         │                                 │  - GitHub     │
         │                                 │  - Base datos │
         └────────────────────────────────►│  - APIs       │
                                           └───────────────┘
```

---

## 2. ¿Para qué sirve?

MCP permite que Claude (u otros modelos) puedan:
- Leer y escribir archivos de tu ordenador
- Acceder a GitHub (repos, issues, PRs)
- Consultar bases de datos (SQLite, PostgreSQL)
- Buscar en internet
- Conectar con Slack, Google Drive, Notion, etc.
- Ejecutar comandos en la terminal
- Interactuar con APIs externas
- **Automatizar navegadores web** con Playwright (navegar, hacer clicks, extraer datos)

**Ejemplo real:**
Con MCP configurado, puedes decirle a Claude Desktop:
> "Lee el archivo informe.pdf de mi escritorio y resúmelo"

Y Claude realmente accede al archivo, no solo te dice cómo hacerlo.

---

## 3. Arquitectura de MCP

### 3.1 Componentes

| Componente | Qué es | Ejemplo |
|------------|--------|---------|
| **Host** | La aplicación cliente | Claude Desktop, VS Code, OpenCode |
| **Server** | El "plugin" que da una capacidad | filesystem, github, sqlite |
| **Protocol** | Cómo se comunican | JSON-RPC 2.0 sobre stdio/HTTP |

### 3.2 Capacidades de un Server

Un MCP Server puede exponer tres tipos de capacidades:

| Capacidad | Descripción | Ejemplo |
|-----------|-------------|---------|
| **Tools** | Funciones que el LLM puede ejecutar | `read_file`, `create_issue` |
| **Resources** | Datos que el LLM puede leer | Lista de archivos, contenido de un repo |
| **Prompts** | Templates predefinidos | "Resumir documento", "Revisar código" |

---

## 4. MCP Servers Populares

### 4.1 Servers Oficiales

| Server | Qué hace | Instalación |
|--------|----------|-------------|
| **@modelcontextprotocol/server-filesystem** | Leer/escribir archivos | `npx @modelcontextprotocol/server-filesystem` |
| **@modelcontextprotocol/server-github** | Repos, issues, PRs | `npx @modelcontextprotocol/server-github` |
| **@modelcontextprotocol/server-sqlite** | Consultar SQLite | `npx @modelcontextprotocol/server-sqlite` |
| **@modelcontextprotocol/server-postgres** | Consultar PostgreSQL | `npx @modelcontextprotocol/server-postgres` |
| **@modelcontextprotocol/server-brave-search** | Buscar en internet | `npx @modelcontextprotocol/server-brave-search` |
| **@modelcontextprotocol/server-memory** | Memoria persistente | `npx @modelcontextprotocol/server-memory` |
| **@playwright/mcp** | Automatización de navegadores | `npx @playwright/mcp` |

### 4.2 Servers de la Comunidad

| Server | Qué hace |
|--------|----------|
| **mcp-server-fetch** | Hacer peticiones HTTP |
| **mcp-server-slack** | Enviar/leer mensajes de Slack |
| **mcp-server-notion** | Acceder a Notion |
| **mcp-server-google-drive** | Acceder a Google Drive |
| **mcp-server-youtube** | Transcribir vídeos |

🔗 **Repositorio oficial**: https://github.com/modelcontextprotocol/servers

---

## 5. Configuración en Claude Desktop

### 5.1 Ubicación del archivo de configuración

| Sistema | Ruta |
|---------|------|
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Mac** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Linux** | `~/.config/Claude/claude_desktop_config.json` |

### 5.2 Ejemplo de configuración básica

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\TuUsuario\\Documents"
      ]
    }
  }
}
```

### 5.3 Ejemplo con múltiples servers

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\TuUsuario\\Documents",
        "C:\\Users\\TuUsuario\\Desktop"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_tu_token_aqui"
      }
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "tu_api_key"
      }
    }
  }
}
```

### 5.4 Pasos para activar

1. **Instala Node.js** (necesario para npx): https://nodejs.org
2. **Cierra Claude Desktop** completamente
3. **Crea/edita** el archivo de configuración
4. **Reinicia Claude Desktop**
5. **Verifica**: deberías ver un icono de herramientas 🔧 en la interfaz

---

## 6. Tutorial: Instalar MCPs en OpenCode

**OpenCode** es un editor de código con IA integrada que soporta MCP. Aquí te explicamos cómo configurarlo.

### 6.1 Requisitos previos

```bash
# 1. Instalar Node.js (versión 18 o superior)
# Descarga desde: https://nodejs.org

# 2. Verificar instalación
node --version   # Debería mostrar v18.x.x o superior
npm --version    # Debería mostrar 9.x.x o superior
```

### 6.2 Ubicación del archivo de configuración

OpenCode usa un archivo de configuración similar a Claude Desktop:

| Sistema | Ruta |
|---------|------|
| **Windows** | `%USERPROFILE%\.opencode\mcp.json` |
| **Mac/Linux** | `~/.opencode/mcp.json` |

### 6.3 Paso a paso: Configurar MCP de Filesystem

**Paso 1**: Crear la carpeta de configuración (si no existe)

```bash
# Windows (PowerShell)
mkdir "$env:USERPROFILE\.opencode" -Force

# Mac/Linux
mkdir -p ~/.opencode
```

**Paso 2**: Crear el archivo `mcp.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/ruta/a/tu/proyecto"
      ]
    }
  }
}
```

**Paso 3**: Reiniciar OpenCode

### 6.4 Paso a paso: Configurar MCP de GitHub

**Paso 1**: Obtener un token de GitHub
1. Ve a https://github.com/settings/tokens
2. Click en "Generate new token (classic)"
3. Selecciona los permisos: `repo`, `read:org`
4. Copia el token generado

**Paso 2**: Añadir al archivo `mcp.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/ruta/a/tu/proyecto"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_tu_token_aqui"
      }
    }
  }
}
```

**Paso 3**: Reiniciar OpenCode

### 6.5 Paso a paso: Configurar MCP de SQLite

**Paso 1**: Asegúrate de tener una base de datos SQLite

```bash
# Crear una base de datos de ejemplo
sqlite3 mi_base.db "CREATE TABLE usuarios (id INTEGER PRIMARY KEY, nombre TEXT);"
```

**Paso 2**: Añadir al archivo `mcp.json`

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "/ruta/a/mi_base.db"
      ]
    }
  }
}
```

### 6.6 Paso a paso: Configurar MCP de Playwright (Automatización Web)

**Playwright** es una herramienta de Microsoft para automatizar navegadores. Con el MCP de Playwright, la IA puede:
- Navegar por páginas web
- Hacer clicks, rellenar formularios
- Tomar capturas de pantalla
- Extraer información de páginas
- Ejecutar pruebas automatizadas

**Paso 1**: Instalar Playwright (si no lo tienes)

```bash
# Instalar Playwright globalmente
npm install -g playwright

# Instalar los navegadores (Chromium, Firefox, WebKit)
npx playwright install
```

**Paso 2**: Añadir al archivo `mcp.json`

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp"]
    }
  }
}
```

**Paso 3**: Reiniciar OpenCode

**Paso 4**: Probar con comandos como:
- "Abre la página de Google y busca 'Python tutorial'"
- "Navega a wikipedia.org y toma una captura de pantalla"
- "Rellena el formulario de contacto en mi web"

#### Opciones avanzadas de Playwright MCP

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp",
        "--browser", "chromium",
        "--headless"
      ]
    }
  }
}
```

| Opción | Descripción |
|--------|-------------|
| `--browser chromium` | Usar Chromium (por defecto) |
| `--browser firefox` | Usar Firefox |
| `--browser webkit` | Usar WebKit (Safari) |
| `--headless` | Ejecutar sin interfaz gráfica |
| `--viewport 1920x1080` | Tamaño de ventana |

#### Herramientas disponibles en Playwright MCP

| Herramienta | Descripción |
|-------------|-------------|
| `browser_navigate` | Navegar a una URL |
| `browser_click` | Hacer click en un elemento |
| `browser_type` | Escribir texto en un campo |
| `browser_screenshot` | Tomar captura de pantalla |
| `browser_get_text` | Obtener texto de la página |
| `browser_wait` | Esperar a que aparezca un elemento |
| `browser_evaluate` | Ejecutar JavaScript en la página |

#### Ejemplo práctico: Web Scraping con IA

Una vez configurado, puedes pedirle a la IA:

```
"Ve a https://quotes.toscrape.com, extrae las primeras 5 citas
y guárdalas en un archivo quotes.txt"
```

La IA usará Playwright para:
1. Abrir el navegador
2. Navegar a la página
3. Extraer el contenido
4. Guardarlo en un archivo (si tienes filesystem MCP)

---

### 6.7 Configuración completa de ejemplo

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Estudiante\\Proyectos"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "C:\\Users\\Estudiante\\datos.db"
      ]
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "mcp-server-fetch"]
    }
  }
}
```

### 6.8 Verificar que funciona

Una vez configurado, abre OpenCode y prueba:

1. **Filesystem**: "Lista los archivos en mi carpeta de proyectos"
2. **GitHub**: "Muestra mis repositorios recientes"
3. **SQLite**: "Describe las tablas de mi base de datos"
4. **Playwright**: "Abre google.com y toma una captura de pantalla"

---

## 7. Solución de Problemas Comunes

### Error: "npx no reconocido"

```bash
# Solución: Instalar Node.js
# Windows: Descargar de https://nodejs.org
# Mac: brew install node
# Linux: sudo apt install nodejs npm
```

### Error: "Server not found"

```bash
# Solución: Verificar que el paquete existe
npx @modelcontextprotocol/server-filesystem --help
```

### Error: "Permission denied" (Acceso denegado)

- Verifica que la ruta en la configuración existe
- Asegúrate de tener permisos de lectura/escritura
- En Windows, usa rutas con `\\` doble

### El server no aparece en la interfaz

1. Verifica que el JSON es válido (usa https://jsonlint.com)
2. Reinicia completamente la aplicación
3. Revisa los logs de la aplicación

### Error: "Browser not found" (Playwright)

```bash
# Solución: Instalar los navegadores de Playwright
npx playwright install

# Si solo necesitas Chromium (más ligero):
npx playwright install chromium
```

### Error: "Playwright timeout" (Playwright)

- La página tarda demasiado en cargar
- Solución: Aumentar el timeout o verificar la conexión a internet
- Algunas páginas bloquean automatizaciones (CAPTCHAs)

---

## 8. Crear tu Propio MCP Server

Para usuarios avanzados, puedes crear tu propio MCP Server en Python:

### 8.1 Instalación

```bash
pip install mcp
```

### 8.2 Ejemplo básico

```python
# mi_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Crear el servidor
server = Server("mi-servidor")

# Definir una herramienta
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="saludar",
            description="Genera un saludo personalizado",
            inputSchema={
                "type": "object",
                "properties": {
                    "nombre": {
                        "type": "string",
                        "description": "Nombre de la persona"
                    }
                },
                "required": ["nombre"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "saludar":
        nombre = arguments["nombre"]
        return [TextContent(type="text", text=f"¡Hola, {nombre}! 👋")]

# Ejecutar el servidor
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 8.3 Usar tu servidor personalizado

```json
{
  "mcpServers": {
    "mi-servidor": {
      "command": "python",
      "args": ["C:\\ruta\\a\\mi_server.py"]
    }
  }
}
```

---

## 9. Por qué Importa MCP

| Sin MCP | Con MCP |
|---------|---------|
| Cada app programa sus integraciones | Un estándar para todos |
| Código duplicado | Reutilizable |
| Difícil de mantener | Fácil de actualizar |
| Pocas integraciones | Ecosistema creciente |
| Solo funciona en una app | Funciona en todas las apps compatibles |

---

## 10. Seguridad

⚠️ **Importante:**
- MCP da acceso real a tu sistema
- Solo configura servers de fuentes confiables
- Limita el acceso a carpetas específicas
- No pongas credenciales en el código (usa variables de entorno)
- Revisa el código de servers de terceros antes de usarlos

### Buenas prácticas de seguridad

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\MiUsuario\\ProyectoSeguro"
      ]
    }
  }
}
```

✅ **Bien**: Limitar a una carpeta específica del proyecto
❌ **Mal**: Dar acceso a `C:\` o `/` (raíz del sistema)

---

## 11. Recursos Adicionales

| Recurso | Enlace |
|---------|--------|
| Documentación oficial | https://modelcontextprotocol.io |
| Repositorio de servers | https://github.com/modelcontextprotocol/servers |
| Especificación del protocolo | https://spec.modelcontextprotocol.io |
| SDK de Python | https://github.com/modelcontextprotocol/python-sdk |
| SDK de TypeScript | https://github.com/modelcontextprotocol/typescript-sdk |

---

## 12. Resumen

- **MCP** = estándar abierto para conectar IAs con herramientas
- **Creado por Anthropic** (los de Claude)
- **Permite**: acceso a archivos, bases de datos, GitHub, APIs, navegadores web, etc.
- **Se configura**: editando un archivo JSON
- **Compatible con**: Claude Desktop, VS Code, OpenCode, y más
- **Extensible**: puedes crear tus propios servers
- **Playwright**: automatiza navegadores para web scraping, testing y más
- **Seguridad**: ten cuidado con qué accesos das
- **Creciendo**: cada vez más servers y aplicaciones compatibles

---

## 13. Ejercicios Prácticos

### Ejercicio 1: Configurar MCP de Filesystem

**Objetivo**: Configurar acceso a archivos locales

1. Instala Node.js si no lo tienes
2. Crea el archivo de configuración en la ubicación correcta
3. Configura acceso a tu carpeta de proyectos del curso
4. Reinicia tu aplicación (Claude Desktop u OpenCode)
5. Prueba pidiendo: "Lista los archivos .py en mi carpeta"

**Entrega**: Captura de pantalla mostrando que el MCP está funcionando correctamente.

---

### Ejercicio 2: Automatización Web con Playwright

**Objetivo**: Usar Playwright MCP para extraer información de una web

1. Configura el MCP de Playwright siguiendo la sección 6.6
2. Instala los navegadores: `npx playwright install chromium`
3. Reinicia tu aplicación
4. Pide a la IA: "Ve a https://quotes.toscrape.com y extrae las primeras 3 citas con sus autores"
5. Opcional: Pide que guarde el resultado en un archivo (necesita filesystem MCP)

**Entrega**: Captura de pantalla del navegador automatizado y el resultado extraído.

---

### Ejercicio 3 (Avanzado): Combinar múltiples MCPs

**Objetivo**: Usar filesystem + playwright juntos

1. Configura ambos MCPs (filesystem y playwright)
2. Pide a la IA: "Busca en Google 'Python tutorial', extrae los 5 primeros resultados y guárdalos en un archivo resultados.txt"
3. Verifica que el archivo se creó correctamente

**Entrega**: El archivo resultados.txt generado.
