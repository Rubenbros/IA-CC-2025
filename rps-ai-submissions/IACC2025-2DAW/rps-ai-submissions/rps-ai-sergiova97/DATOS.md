# Documentacion de Recogida de Datos

**Alumnos:** García Egea, Francisco y Villanueva Agreda, Sergio

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [x] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:
Elaboramos un pequeño programa para jugar contra otra persona. En él recogemos en el CSV, 
```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)

Se creó un programa para jugar partidas 1v1 (piedraPapelTijeras1vs1.py). De cada partida se recogen el número de la ronda, 
la jugada de ambos y su tiempo de respuesta.

También se han recogido datos de otras personas (datos de la clase de DAM2, concreatamente de David García),
se ha intentado juntarlos limpiando formato, todos los csv que se han usado para intentar sacar el mejor modelo
se comparten en este proyecto.

El nuestro propio es el 'general1vs1.csv'. 

```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [x] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [x] `resultado` - victoria/derrota/empate
- [x] Otro: ronda

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)

ronda: En cada ejecución del programa la ronda se resetea a 0 para ver cuantas rondas tiene cada partida

```

---

## Estadisticas del dataset

- **Total de rondas:** 369
- **Numero de sesiones/partidas:** 50 (de media)
- **Contra cuantas personas diferentes:** 5

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [x] **IA General**: Entrenada para ganar a cualquier oponente

---

## Información adicional

Se han revisado varios modelos, todos ellos se comparten, el que mejor consideramos es el que está por defecto.
