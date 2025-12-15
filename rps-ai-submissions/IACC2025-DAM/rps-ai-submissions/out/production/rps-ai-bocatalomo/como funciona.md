# 🔄 Sincronización en el Juego de la Vida

## 📋 Índice
1. [Problema de Concurrencia](#problema-de-concurrencia)
2. [Herramientas de Sincronización](#herramientas-de-sincronización)
3. [Arquitectura de Sincronización](#arquitectura-de-sincronización)
4. [Flujo Temporal Detallado](#flujo-temporal-detallado)
5. [Ejemplo Visual](#ejemplo-visual)

---

## 🎯 Problema de Concurrencia

### El Desafío

Tenemos **401 hilos** ejecutándose en paralelo:
- **400 celdas** (cada una un hilo independiente)
- **1 controlador** (gestiona las generaciones)

### ¿Qué podría salir mal?

```
❌ SIN SINCRONIZACIÓN:

Celda A: Lee vecinos → Calcula → Actualiza ✓
Celda B: Lee vecinos (Celda A ya cambió) → Calcula ✗ → Actualiza
Celda C: Lee vecinos → Actualiza (antes de calcular) ✗

Resultado: Caos total, células aparecen donde no deberían
```

### Requisitos de Sincronización

1. ✅ Todas las celdas deben **calcular ANTES** de que cualquiera actualice
2. ✅ Todas las celdas deben **actualizar AL MISMO TIEMPO**
3. ✅ El controlador debe **esperar** a que todas terminen
4. ✅ Las celdas deben **esperar permiso** del controlador antes de avanzar

---

## 🛠️ Herramientas de Sincronización

### 1️⃣ **CyclicBarrier** (Barrera Cíclica)

**¿Qué es?**
- Punto de encuentro donde los hilos esperan unos a otros
- Se "rompe" cuando llegan todos los participantes
- Es "cíclica" porque se puede reutilizar

**Analogía:**
```
Imagina una carrera de relevos:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ A   │  │ B   │  │ C   │  │ D   │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │
   ▼        ▼        ▼        ▼
   ═════════════════════════════  ← BARRERA
   (Todos esperan aquí)
   
Cuando el último (D) llega:
   ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
   ¡TODOS continúan juntos!
```

**En Java:**
```java
CyclicBarrier barrera = new CyclicBarrier(4); // 4 participantes

// En cada hilo:
barrera.await(); // Esperar aquí hasta que lleguen los 4
// Continuar todos juntos...
```

---

### 2️⃣ **Lock + Condition** (Cerradura + Condición)

**¿Qué es?**
- **Lock**: Como una llave que solo un hilo puede tener
- **Condition**: Permite esperar/despertar hilos bajo ciertas condiciones

**Analogía:**
```
🚪 Puerta con llave (Lock):
- Solo uno puede entrar a la vez
- Otros esperan fuera

📢 Megáfono (Condition):
- Controlador grita: "¡Adelante!"
- Todos los que esperan se despiertan
```

**En Java:**
```java
Lock lock = new ReentrantLock();
Condition condicion = lock.newCondition();
boolean puedeAvanzar = false;

// Esperar:
lock.lock();
try {
    while (!puedeAvanzar) {
        condicion.await(); // Dormir
    }
} finally {
    lock.unlock();
}

// Despertar:
lock.lock();
try {
    puedeAvanzar = true;
    condicion.signalAll(); // ¡Despierten todos!
} finally {
    lock.unlock();
}
```

---

### 3️⃣ **volatile** (Visibilidad)

**¿Qué hace?**
- Garantiza que todos los hilos vean el valor actualizado
- Sin caché local por hilo

**Analogía:**
```
Sin volatile:
Hilo A: variable = 5 (guarda en su caché)
Hilo B: lee variable → ve 0 (lee de su caché)

Con volatile:
Hilo A: variable = 5 (escribe en memoria principal)
Hilo B: lee variable → ve 5 (lee de memoria principal)
```

**En Java:**
```java
private volatile boolean activo = true;
```

---

### 4️⃣ **synchronized** (Exclusión Mutua)

**¿Qué hace?**
- Solo un hilo puede ejecutar el código a la vez
- Los demás esperan

**Analogía:**
```
🚻 Baño público:
- Entra 1 persona → 🔒 Cierra
- Otros esperan fuera
- Sale → 🔓 Abre
- Entra el siguiente
```

**En Java:**
```java
public synchronized boolean estaViva() {
    return estadoActual; // Solo 1 hilo lee a la vez
}
```

---

## 🏗️ Arquitectura de Sincronización

### Niveles de Sincronización

```
┌─────────────────────────────────────────────────────────┐
│ NIVEL 1: Control de Avance (Lock + Condition)          │
│ ├─ Controlador da permiso                              │
│ └─ Celdas esperan señal                                │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ NIVEL 2: Sincronización entre Celdas (Barrera 1)       │
│ ├─ 400 celdas calculan en paralelo                     │
│ └─ Esperan hasta que todas calcularon                  │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ NIVEL 3: Actualización Simultánea (Barrera 2)          │
│ ├─ 400 celdas actualizan en paralelo                   │
│ └─ Esperan hasta que todas actualizaron                │
└─────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────┐
│ NIVEL 4: Sincronización con Controlador (Barrera 3)    │
│ ├─ 400 celdas + 1 controlador = 401 total              │
│ └─ Controlador sabe que todas terminaron               │
└─────────────────────────────────────────────────────────┘
```

---

### Las 3 Barreras

#### **Barrera 1: `barreraCalculo` (400 participantes)**
```java
new CyclicBarrier(400) // Solo celdas
```
**Objetivo:** Garantizar que **todas** las celdas calcularon antes de que **cualquiera** actualice.

```
Celda 1: Calcula ✓ → await() → espera...
Celda 2: Calcula ✓ → await() → espera...
...
Celda 400: Calcula ✓ → await() → ¡LIBERA A TODAS!
```

---

#### **Barrera 2: `barreraActualizacion` (400 participantes)**
```java
new CyclicBarrier(400) // Solo celdas
```
**Objetivo:** Garantizar que **todas** las celdas actualizan **simultáneamente**.

```
Celda 1: Actualiza ✓ → await() → espera...
Celda 2: Actualiza ✓ → await() → espera...
...
Celda 400: Actualiza ✓ → await() → ¡LIBERA A TODAS!
```

---

#### **Barrera 3: `barreraSincronizacion` (401 participantes)**
```java
new CyclicBarrier(401) // Celdas + Controlador
```
**Objetivo:** El Controlador **espera** a que todas terminen antes de continuar.

```
Celda 1: Terminó ✓ → await() → espera...
Celda 2: Terminó ✓ → await() → espera...
...
Celda 400: Terminó ✓ → await() → espera...
Controlador: await() → ¡TODOS LIBERADOS!
```

---

## ⏱️ Flujo Temporal Detallado

### Generación N

```
t=0ms
┌──────────────────────────────────────┐
│ CONTROLADOR                          │
│ permitirAvance()                     │
│ ├─ puedeAvanzar = true               │
│ └─ signalAll() → Despierta celdas    │
└──────────────────────────────────────┘
         │
         ▼
t=1ms
┌──────────────────────────────────────┐
│ CELDAS (400 hilos)                   │
│ Se despiertan del await()            │
└──────────────────────────────────────┘
         │
         ▼
t=2-50ms
┌──────────────────────────────────────┐
│ CÁLCULO (en paralelo)                │
│ Celda 1: Cuenta vecinos → Aplica    │
│ Celda 2: Cuenta vecinos → Aplica    │
│ Celda 3: Cuenta vecinos → Aplica    │
│ ...                                  │
│ Celda 400: Cuenta vecinos → Aplica  │
└──────────────────────────────────────┘
         │
         ▼
t=50ms
┌──────────────────────────────────────┐
│ BARRERA 1: barreraCalculo            │
│ Celda 1: await() → espera...         │
│ Celda 2: await() → espera...         │
│ ...                                  │
│ Celda 399: await() → espera...       │
│ Celda 400: await() → ¡LIBERA!        │
└──────────────────────────────────────┘
         │
         ▼
t=51-60ms
┌──────────────────────────────────────┐
│ ACTUALIZACIÓN (en paralelo)          │
│ Celda 1: estadoActual = siguiente    │
│ Celda 2: estadoActual = siguiente    │
│ Celda 3: estadoActual = siguiente    │
│ ...                                  │
│ Celda 400: estadoActual = siguiente  │
└──────────────────────────────────────┘
         │
         ▼
t=60ms
┌──────────────────────────────────────┐
│ BARRERA 2: barreraActualizacion      │
│ Celda 1: await() → espera...         │
│ Celda 2: await() → espera...         │
│ ...                                  │
│ Celda 399: await() → espera...       │
│ Celda 400: await() → ¡LIBERA!        │
└──────────────────────────────────────┘
         │
         ▼
t=61ms
┌──────────────────────────────────────┐
│ BARRERA 3: barreraSincronizacion     │
│ Celda 1: await() → espera...         │
│ Celda 2: await() → espera...         │
│ ...                                  │
│ Celda 400: await() → espera...       │
│ Controlador: await() → ¡LIBERA!      │
└──────────────────────────────────────┘
         │
         ├─────────────────┬────────────────┐
         ▼                 ▼                ▼
    CONTROLADOR        CELDAS          CELDAS
    continúa           vuelven al      esperan
                       inicio          nueva señal
         │
         ▼
t=62ms
┌──────────────────────────────────────┐
│ CONTROLADOR                          │
│ bloquearAvance()                     │
│ ├─ puedeAvanzar = false              │
│ siguienteGeneracion()                │
│ mostrarTablero()                     │
│ sleep(500ms)                         │
└──────────────────────────────────────┘

t=562ms → Repetir para Generación N+1
```

---

## 📊 Ejemplo Visual: Blinker

### Generación 0 → Generación 1

```
GENERACIÓN 0 (inicial):
. . . . .
. # # # .  ← 3 células horizontales
. . . . .

┌─────────────────────────────────────────────────┐
│ INICIO DE GENERACIÓN 1                          │
└─────────────────────────────────────────────────┘

PASO 1: Controlador da permiso
Controlador: permitirAvance()
           └─> signalAll()

PASO 2: Celdas se despiertan y calculan
Celda [9][9]:  Cuenta vecinos → 1 vecino  → Muere
Celda [9][10]: Cuenta vecinos → 2 vecinos → Vive
Celda [9][11]: Cuenta vecinos → 1 vecino  → Muere
Celda [10][10]: Cuenta vecinos → 3 vecinos → Nace ✨
Celda [8][10]:  Cuenta vecinos → 3 vecinos → Nace ✨
... (otras 395 celdas calculan)

PASO 3: BARRERA 1 - Todas esperan
Celda 1: await() ┐
Celda 2: await() │
...              │ 400 celdas esperan
Celda 399: await()│
Celda 400: await()┘ → ¡Todas continúan!

PASO 4: Todas actualizan SIMULTÁNEAMENTE
Celda [9][9]:  estadoActual = false
Celda [9][10]: estadoActual = true
Celda [9][11]: estadoActual = false
Celda [10][10]: estadoActual = true  ✓
Celda [8][10]:  estadoActual = true  ✓
... (otras 395 actualizan)

PASO 5: BARRERA 2 - Todas esperan
Celda 1: await() ┐
Celda 2: await() │
...              │ 400 celdas esperan
Celda 400: await()┘ → ¡Todas continúan!

PASO 6: BARRERA 3 - Sincronización final
Celda 1: await()    ┐
Celda 2: await()    │
...                 │ 401 esperan (400 + controlador)
Celda 400: await()  │
Controlador: await()┘ → ¡TODOS continúan!

PASO 7: Controlador procesa
Controlador: bloquearAvance()
           siguienteGeneracion() → gen = 1
           mostrarTablero()

GENERACIÓN 1 (resultado):
. . . . .
. . # . .  ← 3 células verticales
. . # . .
. . # . .
. . . . .

✅ Correcto: Blinker osciló de horizontal a vertical
```

---

## 🔐 Sincronización en Lecturas

### Problema: Race Condition

```java
// Celda A pregunta: ¿Celda B está viva?
if (tablero.estaCeldaViva(vecinoFila, vecinoColumna)) {
    contador++;
}

// Mientras tanto, Celda B está actualizando:
estadoActual = estadoSiguiente; // ← Race condition
```

### Solución: `synchronized`

```java
// En Tablero.java:
public synchronized boolean estaCeldaViva(int fila, int columna) {
    return celdas[fila][columna].estaViva();
}

// En Celda.java:
public synchronized boolean estaViva() {
    return estadoActual;
}
```

**Garantía:** Solo **un hilo a la vez** puede leer el estado.

---

## 📝 Resumen

| Herramienta | Uso | Participantes |
|-------------|-----|---------------|
| **Lock + Condition** | Control de avance | 400 celdas |
| **Barrera 1** | Esperar que todas calculen | 400 celdas |
| **Barrera 2** | Esperar que todas actualicen | 400 celdas |
| **Barrera 3** | Sincronizar con Controlador | 401 (400 + 1) |
| **synchronized** | Proteger lectura | Todos |
| **volatile** | Visibilidad de flags | Todos |

---

## ✅ Garantías

1. ✅ **Lecturas consistentes**
2. ✅ **Cálculos correctos**
3. ✅ **Actualizaciones atómicas**
4. ✅ **Control del flujo**
5. ✅ **Sin condiciones de carrera**
6. ✅ **Terminación limpia**