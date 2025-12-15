# Documentacion de Recogida de Datos

**Alumno:** Miguel Olles, Jorge Izquierdo, Jose Conde

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna     | Descripcion                        | Ejemplo |
|-------------|------------------------------------|---------|
| `Num_ronda` | Numero de la ronda (1, 2, 3...)    | 1       |
| `Jugador1`  | Nombre del jugador                 | Miguel  |
| `jugada_j1` | Jugada del jugador 1               | papel   |
| `tiempo_j1` | Tieempo de la jugada del jugador 1 | 2.5     |
| `Jugador2`  | Nombre del jugador                 | papel   |
| `jugada_j2` | Jugada del jugador 2               | papel   |
| `tiempo_j2` | Tieempo de la jugada del jugador 2 | papel   |
| `resultado` | Resultado de la partida            | papel   |

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

```
Lo recogimos con un programa propio, estubimos
 alguna clase probandolo




```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [x] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [x] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
SI pero al final no los usamos


```

---

## Estadisticas del dataset

- **Total de rondas:** 139
- **Numero de sesiones/partidas:** 6
- **Contra cuantas personas diferentes:** 4

### Tipo de IA:

- [x] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
