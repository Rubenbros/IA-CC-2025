import java.util.concurrent.BrokenBarrierException;

/**
 * Hilo controlador que gestiona las generaciones del Juego de la Vida.
 * Sincroniza todas las celdas y controla el avance de la simulación.
 */
public class ControladorJuego extends Thread {
    private final Tablero tablero;
    private final int numGeneraciones;
    private final long delayMillis;

    public ControladorJuego(Tablero tablero, int numGeneraciones, long delayMillis) {
        this.tablero = tablero;
        this.numGeneraciones = numGeneraciones;
        this.delayMillis = delayMillis;
    }

    @Override
    public void run() {
        try {
            // Mostrar estado inicial
            System.out.println("\n════════════════════════════════════════════════");
            System.out.println("  JUEGO DE LA VIDA DE CONWAY");
            System.out.println("  Implementación Concurrente con Hilos");
            System.out.println("════════════════════════════════════════════════");
            tablero.mostrarTablero(0);
            Thread.sleep(delayMillis);

            // Ejecutar generaciones
            for (int generacion = 1; generacion <= numGeneraciones; generacion++) {
                // Fase 1: Esperar a que todas las celdas calculen su siguiente estado
                tablero.getBarrera().await();

                // Fase 2: Esperar a que todas las celdas actualicen su estado
                tablero.getBarrera().await();

                // Mostrar el tablero actualizado
                tablero.mostrarTablero(generacion);

                // Pausa entre generaciones
                if (generacion < numGeneraciones) {
                    Thread.sleep(delayMillis);
                }
            }

            // Finalizar simulación
            System.out.println("\n════════════════════════════════════════════════");
            System.out.println("  SIMULACIÓN COMPLETADA");
            System.out.println("  Total de generaciones: " + numGeneraciones);
            System.out.println("════════════════════════════════════════════════\n");

            tablero.detenerSimulacion();

        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
            System.err.println("Error en el controlador: " + e.getMessage());
        }
    }
}