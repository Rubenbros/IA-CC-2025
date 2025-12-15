# Documentacion de Recogida de Datos

**Alumno:** Martin Miroslavov y Carlos Arilla

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

- [X] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)

Cree una función en el código actual que haga una partida de piedra, papel y tijera, hasta 50 rondas. Una vez termine
se guardan en el archivo partidas.csv. Actualmente está puesto para que juegue la IA, pero antes lo cambié y edité el
código para que pueda elegir las dos opciones y así jugar contra mi amigo.




```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [ ] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)


```

---

## Estadisticas del dataset

- **Total de rondas:** 102
- **Numero de sesiones/partidas:** 6
- **Contra cuantas personas diferentes:** 1

### Tipo de IA:

- [X] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Ruben Bros
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
