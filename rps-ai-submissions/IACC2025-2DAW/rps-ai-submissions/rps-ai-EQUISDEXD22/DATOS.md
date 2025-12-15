# Documentacion de Recogida de Datos

**Alumno:** Marcos Uriel Rodíguez

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
- [X] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [X] **Otro**: Parte de los datos son de internet de partidas reales

### Descripcion del proceso:

```
La mayoria de los datos los saque jugando contra mi familia, pero luego
utilice datos de otra gente para añadir mas "realismo" a los datos.

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

- **Total de rondas:** 131
- **Numero de sesiones/partidas:** No lo se exactamente, por mi parte mas de 5
- **Contra cuantas personas diferentes:** No lo se exactamente, por mi parte mas de 5

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [X] **IA General**: Entrenada para ganar a cualquier oponente
