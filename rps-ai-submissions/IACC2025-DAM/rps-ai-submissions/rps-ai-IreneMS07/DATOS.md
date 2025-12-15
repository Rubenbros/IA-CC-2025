# Documentacion de Recogida de Datos

**Alumno:** Irene Martinez y Sergio Gascon

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
Nuestras partidas:
numero_ronda,jugada_j1,jugada_j2
1,piedra,tijera
2,papel,piedra
3,tijera,papel
4,piedra,papel
5,tijera,piedra
6,piedra,papel
7,papel,piedra
8,tijera,piedra
9,piedra,papel
10,tijera,tijera
11,tijera,piedra
12,piedra,papel
13,papel,papel
14,piedra,tijera
15,papel,piedra
16,piedra,piedra
17,piedra,papel
18,papel,tijera
19,piedra,papel
20,tijera,tijera
21,papel,piedra
22,tijera,papel
23,tijera,piedra
24,piedra,piedra
25,papel,papel
26,piedra,tijera
27,piedra,piedra
28,piedra,piedra
29,papel,tijera
30,tijera,tijera
31,piedra,papel
32,tijera,piedra
33,piedra,papel
34,tijera,piedra
35,papel,piedra
36,tijera,tijera
37,piedra,papel
38,tijera,piedra
39,tijera,tijera
40,papel,papel
41,tijera,piedra
42,piedra,piedra
43,papel,tijera
44,tijera,piedra
45,piedra,papel
46,tijera,tijera
47,papel,piedra
48,piedra,tijera
49,tijera,papel
50,piedra,tijera
51,tijera,piedra
52,piedra,tijera
53,papel,piedra
54,tijera,tijera
55,papel,piedra
56,papel,tijera
57,tijera,tijera
58,piedra,piedra
59,papel,papel
60,piedra,papel
61,tijera,piedra
62,piedra,papel
63,tijera,piedra
64,tijera,piedra
65,tijera,piedra
66,tijera,tijera
67,papel,piedra
68,tijera,piedra
69,tijera,piedra
70,piedra,tijera
71,piedra,tijera
72,papel,piedra
73,tijera,tijera
74,papel,piedra
75,tijera,tijera
76,papel,piedra
77,piedra,tijera
78,tijera,papel
79,piedra,tijera
80,papel,tijera
81,piedra,tijera
82,papel,tijera
83,piedra,tijera
84,tijera,piedra
85,papel,tijera
86,tijera,papel
87,piedra,papel
88,piedra,piedra
89,papel,papel
90,tijera,piedra
91,papel,piedra
92,tijera,tijera
93,papel,tijera
94,piedra,tijera
95,tijera,tijera
96,piedra,piedra
97,papel,piedra
98,tijera,tijera
99,piedra,papel
100,tijera,tijera
101,tijera,tijera
102,papel,tijera
103,tijera,piedra
104,papel,piedra
105,piedra,piedra
106,tijera,tijera
107,papel,tijera
108,tijera,piedra
109,papel,piedra
110,papel,papel
111,tijera,papel
112,papel,papel
113,papel,tijera
114,piedra,piedra
115,piedra,papel
---

## Como recogiste los datos?
Cada jugador usaba las teclas qwe y asd para representar piedra, papel y tijera respectivamente

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [ ] **Programa propio**: Cree un programa para jugar y guardar datos
- [x] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:
Cada jugador dice la jugada y se apunta en una hoja de cálculo.
```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)




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

- **Total de rondas:** 115
- **Numero de sesiones/partidas:** 2
- **Contra cuantas personas diferentes:** 100 entre Irene y Sergio y 15 entre Irene y Rubén Jarne

### Tipo de IA:

- [x] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
