# Documentacion de Recogida de Datos

**Alumno:** Lilja Svarva Antonsen

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

Ejemplo del archivo:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `game_number` | Número de la ronda/juego | 3 |
| `p1_last_move` | Última jugada del jugador 1 | P |
| `p1_last_2_moves` | Últimas 2 jugadas del jugador 1 | RP |
| `p1_last_3_moves` | Últimas 3 jugadas del jugador 1 | RRP |
| `p1_rock_freq` | Frecuencia de usar "piedra" jugador 1 | 0.6667 |
| `p1_paper_freq` | Frecuencia de usar "papel" jugador 1 | 0.3333 |
| `p1_scissors_freq` | Frecuencia de usar "tijera" jugador 1 | 0.0 |
| `p1_win_streak` | Racha de victorias jugador 1 | 1 |
| `p1_loss_streak` | Racha de derrotas jugador 1 | 0 |
| `p1_last_outcome` | Resultado de la última jugada del jugador 1 | win |
| `p2_last_move` | Última jugada del jugador 2 | R |
| `p2_last_2_moves` | Últimas 2 jugadas del jugador 2 | PR |
| `p2_last_3_moves` | Últimas 3 jugadas del jugador 2 | RPR |
| `p2_rock_freq` | Frecuencia de usar "piedra" jugador 2 | 0.6667 |
| `p2_paper_freq` | Frecuencia de usar "papel" jugador 2 | 0.3333 |
| `p2_scissors_freq` | Frecuencia de usar "tijera" jugador 2 | 0.0 |
| `p1_current_move` | Jugada actual del jugador 1 | S |


### Ejemplo de CSV:

```csv
game_number,p1_last_move,p1_last_2_moves,p1_last_3_moves,p1_rock_freq,p1_paper_freq,p1_scissors_freq,p1_win_streak,p1_loss_streak,p1_last_outcome,p2_last_move,p2_last_2_moves,p2_last_3_moves,p2_rock_freq,p2_paper_freq,p2_scissors_freq,p1_current_move
3,P,RP,RRP,0.6667,0.3333,0.0,1,0,win,R,PR,RPR,0.6667,0.3333,0.0,S
4,S,PS,RPS,0.5,0.25,0.25,0,0,tie,S,RS,PRS,0.5,0.25,0.25,R
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
Los datos fueron recogidos mediante un programa propio que registraba cada jugada durante partidas de "piedra, papel o tijera". 
Jugué personalmente contra amigos y observé partidas entre ellos para recopilar distintos patrones de juego. 
Una pequeña porción de los datos fue obtenida de partidas jugadas online. 
Cada ronda se registraba con el número de ronda, las jugadas de ambos jugadores y, opcionalmente, estadísticas adicionales como historial de movimientos y rachas.


```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [x] `game_number` - Número de la ronda/juego
- [x] `p1_last_move` - Última jugada del jugador 1
- [x] `p1_last_2_moves` - Últimas 2 jugadas del jugador 1
- [x] `p1_last_3_moves` - Últimas 3 jugadas del jugador 1
- [x] `p1_rock_freq` - Frecuencia de usar "piedra" jugador 1
- [x] `p1_paper_freq` - Frecuencia de usar "papel" jugador 1
- [x] `p1_scissors_freq` - Frecuencia de usar "tijera" jugador 1
- [x] `p1_win_streak` - Racha de victorias jugador 1
- [x] `p1_loss_streak` - Racha de derrotas jugador 1
- [x] `p1_last_outcome` - Resultado de la última jugada del jugador 1 (victoria/derrota/empate)
- [x] `p2_last_move` - Última jugada del jugador 2
- [x] `p2_last_2_moves` - Últimas 2 jugadas del jugador 2
- [x] `p2_last_3_moves` - Últimas 3 jugadas del jugador 2
- [x] `p2_rock_freq` - Frecuencia de usar "piedra" jugador 2
- [x] `p2_paper_freq` - Frecuencia de usar "papel" jugador 2
- [x] `p2_scissors_freq` - Frecuencia de usar "tijera" jugador 2
- [x] `p1_current_move` - Jugada actual del jugador 1



### Descripción de datos adicionales:

```

Se capturaron estadísticas adicionales para analizar patrones de juego y comportamiento estratégico de los jugadores. 
Esto incluye historial de jugadas recientes, frecuencias de movimientos y rachas de victorias/derrotas. 
Estos datos permiten crear modelos predictivos o entrenar IA para anticipar jugadas futuras.

```

### Descripción de tests adicionales:

```
Debido a que me preocupaba la posibilidad de data leakage, realicé pruebas para detectarla en el dataset. 
Como cada vez que recogía nuevos datos los agregaba al CSV existente, utilicé un script para limpiar el archivo y 
asegurarme de que no hubiera entradas duplicadas o redundantes. De esta manera, mantuve la integridad del dataset 
y evité que información previa pudiera influir indebidamente en los análisis o modelos entrenados.


```
---

## Estadisticas del dataset

- **Total de rondas:** 578
- **Numero de sesiones/partidas:** normalmente entre 20 a 50 rondas por partida
- **Contra cuantas personas diferentes:** Al menos 1 oponente principal, más observaciones de amigos

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [x] **IA General**: Entrenada para ganar a cualquier oponente
