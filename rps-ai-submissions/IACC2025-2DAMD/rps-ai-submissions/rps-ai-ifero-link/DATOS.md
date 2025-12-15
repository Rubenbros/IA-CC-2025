# Documentacion de Recogida de Datos

**Alumno:** Irene Fernández Romeo

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
- [x] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
Para el programa propio utilicé la libreria pygame para poder tener dos inputs la mismo tiempo y poder 
tener tambien una ventana de juego mas cómoda de ver que el terminal.

Para el desarrollo simplemente registre la ronda, las jugadas y si habia un cambio de jugada.

Para poder jugar un usuario usaba las teclas ASD y el otro usuario utilizaba JKL la accion de cada 
botón se guarda usando un diccionario y se determina quien a ganado gracias a ña función determinar_resultado.

El CSV se guarda en otra carpeta ya que se pretendian hacer varia partidas y solo se quería meter un 
archivo en la carpeta data.

La parte visual se ha hecho tambien utilizando pygame.

PAra la recogida de datos manual se ha usado un csv local en el que se han ido apuntando los datos 
ha medida que se jugaba. Estos datos se han tenido que recoger de esta manera y no usando el
código creado porque se han recogido datos jugando en linea.

```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [X] `resultado` - victoria/derrota/empate
- [X] Otro: - Si ha habido cambios respecto a la jugada anterior.
```
No se han recogido datos como tiempo_reaccion o timestamp ya que como se han recogido datos de 
diferentes formas hubiera sido contraproducente.
```

### Descripcion de datos adicionales:

```
He recogido estos datos ya que es mas facil comprobar o detectar patrones de si cambia siempre o no
y luego se pueden desarrollar features de si cambia cada vez que empata o gana o si pierde.
```

---

## Estadisticas del dataset

- **Total de rondas:** 385 
- **Numero de sesiones/partidas:** 8
- **Contra cuantas personas diferentes:** 4
```
Aunque se recogieron mas datos al final se solo se usaron 357, que que el porcentaje baja bastante 
de 52 a 41 de accuracy, se ha puesto la combiación de datos que da el mayor porcentaje posible.
Para poder entrenar al modelo.
```

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [X] **IA General**: Entrenada para ganar a cualquier oponente
