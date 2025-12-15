# Documentacion de Recogida de Datos

**Alumno:** Jhonattan Elías Prieto Morillo

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
Hice un juego en Python (src/juego.py) para jugar con otra persona, el juego inicia un contador, cuando termina, los jugadores tienen 3 segundos para jugar.
Se guarda la primera jugada (la opción) elegida de cada jugador y se pasa a la siguiente ronda.
```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [X] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [X] `timestamp` - Fecha/hora de cada jugada
- [X] `sesion` - ID de sesion de juego
- [X] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
-El tiempo de reacción lo considero útil para ver si tenemos jugadas calculadas o impulsivas/aleatorias, también sirvió para comprobar el contador de límite de tiempo por partida.
-La fecha de la partida está bien a modo de seguimiento del proyecto.
-Con la sesión puedo dividir los datos según la persona con la que he probado el juego.
-Resultado: A modo de resumen de cada partida.
```

---

## Estadisticas del dataset

- **Total de rondas:** 203
- **Numero de sesiones/partidas:** 4
- **Contra cuantas personas diferentes:** 4

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [X] **IA General**: Entrenada para ganar a cualquier oponente
