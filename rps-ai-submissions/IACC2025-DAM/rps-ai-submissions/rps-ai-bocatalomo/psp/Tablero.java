import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.Random;

/**
 * Representa el tablero del Juego de la Vida.
 * Gestiona las celdas y la sincronización entre ellas.
 */
public class Tablero {
    private static final int FILAS = 20;
    private static final int COLUMNAS = 20;

    private Celda[][] celdas;
    private CyclicBarrier barreraCalculo;
    private CyclicBarrier barreraActualizacion;
    private CyclicBarrier barreraSincronizacion;  // Nueva: Controlador + Celdas
    private int generacion;

    /**
     * Constructor del Tablero
     */
    public Tablero() {
        this.generacion = 0;
        this.celdas = new Celda[FILAS][COLUMNAS];

        // ════════════════════════════════════════════════════════════
        // BARRERA 1: Solo celdas - Cuando todas calculan
        // ════════════════════════════════════════════════════════════
        this.barreraCalculo = new CyclicBarrier(FILAS * COLUMNAS);

        // ════════════════════════════════════════════════════════════
        // BARRERA 2: Solo celdas - Cuando todas actualizan
        // ════════════════════════════════════════════════════════════
        this.barreraActualizacion = new CyclicBarrier(FILAS * COLUMNAS);

        // ════════════════════════════════════════════════════════════
        // BARRERA 3: Celdas + Controlador (401 participantes)
        // Sincroniza el final de cada generación
        // ════════════════════════════════════════════════════════════
        this.barreraSincronizacion = new CyclicBarrier(FILAS * COLUMNAS + 1);

        // Crear todas las celdas
        inicializarCeldas();
    }

    /**
     * Crea todas las celdas del tablero
     */
    private void inicializarCeldas() {
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                celdas[i][j] = new Celda(i, j, this, barreraCalculo,
                        barreraActualizacion, barreraSincronizacion);
            }
        }
    }

    /**
     * Inicializa el tablero con un patrón aleatorio
     */
    public void inicializarAleatorio(double densidad) {
        Random random = new Random();
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                boolean viva = random.nextDouble() < densidad;
                celdas[i][j].setEstado(viva);
            }
        }
    }

    /**
     * Inicializa el tablero con un patrón predefinido
     */
    public void inicializarConPatron(String patron) {
        // Limpiar tablero primero
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                celdas[i][j].setEstado(false);
            }
        }

        // Posición central del tablero
        int centroFila = FILAS / 2;
        int centroColumna = COLUMNAS / 2;

        switch (patron.toLowerCase()) {
            case "blinker":
                // Oscilador periodo 2 (horizontal)
                celdas[centroFila][centroColumna - 1].setEstado(true);
                celdas[centroFila][centroColumna].setEstado(true);
                celdas[centroFila][centroColumna + 1].setEstado(true);
                break;

            case "glider":
                // Se mueve diagonalmente
                celdas[centroFila - 1][centroColumna].setEstado(true);
                celdas[centroFila][centroColumna + 1].setEstado(true);
                celdas[centroFila + 1][centroColumna - 1].setEstado(true);
                celdas[centroFila + 1][centroColumna].setEstado(true);
                celdas[centroFila + 1][centroColumna + 1].setEstado(true);
                break;

            case "toad":
                // Oscilador periodo 2
                celdas[centroFila][centroColumna].setEstado(true);
                celdas[centroFila][centroColumna + 1].setEstado(true);
                celdas[centroFila][centroColumna + 2].setEstado(true);
                celdas[centroFila + 1][centroColumna - 1].setEstado(true);
                celdas[centroFila + 1][centroColumna].setEstado(true);
                celdas[centroFila + 1][centroColumna + 1].setEstado(true);
                break;

            case "block":
                // Naturaleza muerta (no cambia)
                celdas[centroFila][centroColumna].setEstado(true);
                celdas[centroFila][centroColumna + 1].setEstado(true);
                celdas[centroFila + 1][centroColumna].setEstado(true);
                celdas[centroFila + 1][centroColumna + 1].setEstado(true);
                break;

            default:
                System.err.println("Patrón desconocido: " + patron);
                inicializarAleatorio(0.25);
        }
    }

    /**
     * Inicia la simulación arrancando todos los hilos de celdas
     */
    public void iniciarSimulacion() {
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                celdas[i][j].start();
            }
        }
    }

    /**
     * Detiene todos los hilos de celdas
     */
    public void detenerSimulacion() {
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                celdas[i][j].detener();
            }
        }
    }

    /**
     * Verifica si una posición está dentro del tablero
     */
    public boolean dentroDelTablero(int fila, int columna) {
        return fila >= 0 && fila < FILAS && columna >= 0 && columna < COLUMNAS;
    }

    /**
     * Consulta si una celda está viva (SINCRONIZADO)
     */
    public synchronized boolean estaCeldaViva(int fila, int columna) {
        return celdas[fila][columna].estaViva();
    }

    /**
     * Obtiene una celda (para debugging)
     */
    public Celda getCelda(int fila, int columna) {
        return celdas[fila][columna];
    }

    /**
     * Verifica si el tablero está vacío (todas muertas)
     */
    public boolean estaVacio() {
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                if (celdas[i][j].estaViva()) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Cuenta cuántas celdas están vivas
     */
    private int contarCeldasVivas() {
        int contador = 0;
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                if (celdas[i][j].estaViva()) {
                    contador++;
                }
            }
        }
        return contador;
    }

    /**
     * Incrementa el contador de generación
     */
    public void siguienteGeneracion() {
        generacion++;
    }

    /**
     * Obtiene el número de generación actual
     */
    public int getGeneracion() {
        return generacion;
    }

    /**
     * Obtiene la barrera de sincronización (para el Controlador)
     */
    public CyclicBarrier getBarreraSincronizacion() {
        return barreraSincronizacion;
    }

    /**
     * Muestra el tablero en consola
     */
    public void mostrarTablero() {
        limpiarConsola();

        System.out.println("\n+============================================+");
        System.out.println("|   JUEGO DE LA VIDA - Generacion: " +
                String.format("%3d", generacion) + "     |");
        System.out.println("+============================================+");

        for (int i = 0; i < FILAS; i++) {
            System.out.print("| ");
            for (int j = 0; j < COLUMNAS; j++) {
                if (celdas[i][j].estaViva()) {
                    System.out.print("# ");
                } else {
                    System.out.print(". ");
                }
            }
            System.out.println("|");
        }

        int vivas = contarCeldasVivas();
        int muertas = (FILAS * COLUMNAS) - vivas;

        System.out.println("+============================================+");
        System.out.println("| Vivas: " + String.format("%3d", vivas) +
                " | Muertas: " + String.format("%3d", muertas) +
                " | Total: " + (FILAS * COLUMNAS) + "   |");
        System.out.println("+============================================+\n");
    }

    /**
     * Limpia la consola
     */
    private void limpiarConsola() {
        try {
            if (System.getProperty("os.name").contains("Windows")) {
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            } else {
                System.out.print("\033[H\033[2J");
                System.out.flush();
            }
        } catch (Exception e) {
            // Fallback: imprimir líneas vacías
            for (int i = 0; i < 50; i++) {
                System.out.println();
            }
        }
    }
}