public class JuegoVida {

    // ========== CONFIGURACIÓN DEL TABLERO ==========
    /**
     * Número de filas del tablero
     */
    private static final int FILAS = 20;

    /**
     * Número de columnas del tablero
     */
    private static final int COLUMNAS = 30;

    /**
     * Probabilidad de que una celda esté viva al inicio (0.0 - 1.0)
     * 0.3 = 30% de celdas vivas
     */
    private static final double PROBABILIDAD_VIDA_INICIAL = 0.15;

    // ========== CONFIGURACIÓN DE LA SIMULACIÓN ==========
    /**
     * Número total de generaciones a simular
     */
    private static final int NUM_GENERACIONES = 10;

    /**
     * Tiempo de espera entre generaciones en milisegundos
     * 500ms = 0.5 segundos
     */
    private static final long DELAY_ENTRE_GENERACIONES = 1000;

    /**
     * Método principal que inicia la simulación
     */
    public static void main(String[] args) {
        System.out.println("Iniciando simulación del Juego de la Vida...");
        System.out.println("Tablero: " + FILAS + "x" + COLUMNAS);
        System.out.println("Generaciones: " + NUM_GENERACIONES);
        System.out.println("Delay: " + DELAY_ENTRE_GENERACIONES + "ms\n");

        // Crear el tablero con la configuración especificada
        Tablero tablero = new Tablero(FILAS, COLUMNAS, PROBABILIDAD_VIDA_INICIAL);

        // Iniciar todos los hilos de las celdas
        tablero.iniciarSimulacion();

        // Crear y arrancar el hilo controlador
        ControladorJuego controlador = new ControladorJuego(
                tablero,
                NUM_GENERACIONES,
                DELAY_ENTRE_GENERACIONES
        );
        controlador.start();

        // Esperar a que el controlador termine
        try {
            controlador.join();
            System.out.println("Programa finalizado correctamente.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Simulación interrumpida: " + e.getMessage());
        }
    }
}