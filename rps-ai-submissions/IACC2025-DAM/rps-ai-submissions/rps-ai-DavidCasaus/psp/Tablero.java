import java.util.concurrent.CyclicBarrier;

/**
 * Representa el tablero del Juego de la Vida.
 * Gestiona la matriz de celdas y su sincronización.
 */
public class Tablero {
    private final int filas;
    private final int columnas;
    private final Celda[][] celdas;
    private final CyclicBarrier barrera;

    public Tablero(int filas, int columnas, double probabilidadVida) {
        this.filas = filas;
        this.columnas = columnas;
        this.celdas = new Celda[filas][columnas];

        // La barrera se activa cuando todas las celdas llegan + 1 para el controlador
        this.barrera = new CyclicBarrier(filas * columnas + 1);

        inicializarTablero(probabilidadVida);
    }

    /**
     * Inicializa el tablero con celdas en estado aleatorio
     */
    private void inicializarTablero(double probabilidadVida) {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                boolean estadoInicial = Math.random() < probabilidadVida;
                celdas[i][j] = new Celda(i, j, estadoInicial, this, barrera);
            }
        }
    }

    /**
     * Inicia todos los hilos de las celdas
     */
    public void iniciarSimulacion() {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                celdas[i][j].start();
            }
        }
    }

    /**
     * Detiene todos los hilos de las celdas
     */
    public void detenerSimulacion() {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                celdas[i][j].detener();
            }
        }

        // Esperar a que todos los hilos terminen
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                try {
                    celdas[i][j].join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    /**
     * Obtiene el estado de una celda específica
     */
    public boolean obtenerEstadoCelda(int fila, int columna) {
        return celdas[fila][columna].estaViva();
    }

    /**
     * Devuelve la barrera de sincronización
     */
    public CyclicBarrier getBarrera() {
        return barrera;
    }

    /**
     * Muestra el estado actual del tablero en consola
     */
    public void mostrarTablero(int generacion) {
        System.out.println("\n╔" + "═".repeat(columnas * 2) + "╗");
        System.out.println("║ GENERACIÓN: " + generacion + " ".repeat(Math.max(0, columnas * 2 - 14 - String.valueOf(generacion).length())) + "║");
        System.out.println("╠" + "═".repeat(columnas * 2) + "╣");

        for (int i = 0; i < filas; i++) {
            System.out.print("║");
            for (int j = 0; j < columnas; j++) {
                System.out.print(celdas[i][j].estaViva() ? "██" : "  ");
            }
            System.out.println("║");
        }

        System.out.println("╚" + "═".repeat(columnas * 2) + "╝");
    }

    public int getFilas() {
        return filas;
    }

    public int getColumnas() {
        return columnas;
    }
}