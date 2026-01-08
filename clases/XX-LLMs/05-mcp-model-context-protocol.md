# MCP (Model Context Protocol)

## 1. ¿Qué es MCP?

**MCP** es un estándar creado por Anthropic (los de Claude) para conectar IAs con herramientas externas de forma universal.

**Problema que resuelve:**
Antes, cada aplicación tenía que programar sus propias conexiones con herramientas. Con MCP, se hace una vez y funciona para todas.

**Analogía:** MCP es como el USB. Antes cada dispositivo tenía su propio conector, ahora con USB todo se conecta igual.

---

## 2. ¿Para qué sirve?

MCP permite que Claude (u otros modelos) puedan:
- Leer y escribir archivos de tu ordenador
- Acceder a GitHub
- Consultar bases de datos
- Buscar en internet
- Conectar con Slack, Google Drive, etc.

**Ejemplo real:**
Con MCP configurado, puedes decirle a Claude Desktop:
> "Lee el archivo informe.pdf de mi escritorio y resúmelo"

Y Claude realmente accede al archivo, no solo te dice cómo hacerlo.

---

## 3. Componentes

| Componente | Qué es |
|------------|--------|
| **Host** | La aplicación (ej: Claude Desktop) |
| **Server** | El "plugin" que da una capacidad (ej: acceso a archivos) |
| **Protocol** | Cómo se comunican (mensajes JSON) |

---

## 4. MCP Servers Disponibles

Algunos servidores MCP oficiales:

| Server | Qué hace |
|--------|----------|
| **Filesystem** | Leer/escribir archivos locales |
| **GitHub** | Acceder a repos, issues, PRs |
| **SQLite** | Consultar bases de datos |
| **Brave Search** | Buscar en internet |
| **Google Drive** | Acceder a documentos en la nube |
| **Slack** | Enviar mensajes |

---

## 5. Cómo Usar MCP (Claude Desktop)

1. **Descarga Claude Desktop** desde claude.ai/download
2. **Edita el archivo de configuración:**
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. **Añade los servers** que quieras usar
4. **Reinicia Claude Desktop**

---

## 6. Por qué Importa MCP

| Sin MCP | Con MCP |
|---------|---------|
| Cada app programa sus integraciones | Un estándar para todos |
| Código duplicado | Reutilizable |
| Difícil de mantener | Fácil de actualizar |
| Pocas integraciones | Ecosistema creciente |

---

## 7. Seguridad

⚠️ **Importante:**
- MCP da acceso real a tu sistema
- Solo configura servers de fuentes confiables
- Limita el acceso a carpetas específicas
- No pongas credenciales en el código

---

## 8. Resumen

- **MCP** = estándar para conectar IAs con herramientas
- **Creado por Anthropic** (los de Claude)
- **Permite**: acceso a archivos, bases de datos, APIs, etc.
- **Se configura en Claude Desktop** editando un JSON
- **Creciendo**: cada vez más servers disponibles
- **Seguridad**: ten cuidado con qué accesos das

