# Documentacion de Recogida de Datos

**Alumno:** _Rodrigo Vladimir Alcántara Gutierrez__

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

- [ ] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [X] **Otro**: _Guarda los datos internamente segun se va jugando la partida_
### Descripcion del proceso:

```
Los datos en los que se basa es en un modelo que se llama "Cadena de Markov" el cuál lee
la/s jugadas anteriores para poder predecir la siguiente jugada de la persona y que tengas 
más probabilidades de ganar, guardando en una tabla "mental" las jugadas que ha ido haciendo
el jugador.  




```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [X] `resultado` - victoria/derrota/empate
- [X] Otro: El número de ronda en el que están, tu jugada, la jugada de la IA y su eficiencia,
además del número total de victorias, derrotas y empates de la IA.

### Descripcion de datos adicionales:

```
Capturo estos datos para que la persona que juega sepa en que ronda se encuentra,
además de tener un cálculo para saber cuantas veces ha ganado la IA, además de al final
del total de partidas darte un porcetaje con la eficiencia de la IA.


```

---

## Estadisticas del dataset

- **Total de rondas:** __50__
- **Numero de sesiones/partidas:** __25__
- **Contra cuantas personas diferentes:** __1__

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [X] **IA General**: Entrenada para ganar a cualquier oponente
