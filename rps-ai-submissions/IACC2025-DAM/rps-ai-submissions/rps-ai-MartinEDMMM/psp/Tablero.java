public class Tablero {
    private final int FILAS = 10;
    private final int COLUMNAS = 10;
    private boolean[][] mundo;

    public Tablero() {
        mundo = new boolean[FILAS][COLUMNAS];
    }

    /**
     * Establece una célula como viva en la posición (fila, columna).
     */
    public void establecerVida(int fila, int columna) {
        if (fila >= 0 && fila < FILAS && columna >= 0 && columna < COLUMNAS) {
            mundo[fila][columna] = true;
        }
    }

    /**
     * Calcula y aplica las reglas para la siguiente generación.
     */
    public void siguienteGeneracion() {
        boolean[][] nuevoMundo = new boolean[FILAS][COLUMNAS];

        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                int vecinosVivos = contarVecinosVivos(i, j);

                // 1. Regla de Supervivencia:
                if (mundo[i][j]) { // Si la célula está viva
                    if (vecinosVivos < 2) {
                        nuevoMundo[i][j] = false; // Muere por soledad
                    } else if (vecinosVivos == 2 || vecinosVivos == 3) {
                        nuevoMundo[i][j] = true; // Sobrevive
                    } else if (vecinosVivos > 3) {
                        nuevoMundo[i][j] = false; // Muere por superpoblación
                    }
                }
                // 2. Regla de Nacimiento:
                else { // Si la célula está muerta
                    if (vecinosVivos == 3) {
                        nuevoMundo[i][j] = true; // Nace una nueva vida
                    }
                }
            }
        }
        mundo = nuevoMundo;
    }

    /**
     * Cuenta cuántos vecinos vivos tiene una célula en (fila, columna).
     */
    private int contarVecinosVivos(int fila, int columna) {
        int cuenta = 0;
        // Recorre las 8 posiciones vecinas
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // Evita contar la propia célula
                if (i == 0 && j == 0) continue;

                int vecinoFila = fila + i;
                int vecinoColumna = columna + j;

                // Comprueba que la posición del vecino esté dentro de los límites del tablero
                if (vecinoFila >= 0 && vecinoFila < FILAS && vecinoColumna >= 0 && vecinoColumna < COLUMNAS) {
                    if (mundo[vecinoFila][vecinoColumna]) {
                        cuenta++;
                    }
                }
            }
        }
        return cuenta;
    }

    /**
     * Muestra el estado actual del tablero en la consola.
     */
    public void imprimirTablero() {
        System.out.println("--- Generación Actual ---");
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                // '*' para célula viva, '.' para célula muerta
                System.out.print(mundo[i][j] ? " * " : " . ");
            }
            System.out.println();
        }
        System.out.println("-------------------------");
    }
}