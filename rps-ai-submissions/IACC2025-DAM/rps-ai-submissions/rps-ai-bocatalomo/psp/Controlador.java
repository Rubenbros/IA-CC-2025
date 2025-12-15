import java.util.concurrent.BrokenBarrierException;

/**
 * Controla el avance de las generaciones del Juego de la Vida.
 */
public class Controlador extends Thread {
    private Tablero tablero;
    private int numeroGeneraciones;
    private int delayMilisegundos;

    public Controlador(Tablero tablero, int numeroGeneraciones, int delayMilisegundos) {
        this.tablero = tablero;
        this.numeroGeneraciones = numeroGeneraciones;
        this.delayMilisegundos = delayMilisegundos;
    }

    @Override
    public void run() {
        try {
            // ════════════════════════════════════════════════
            // ESTADO INICIAL (Generación 0)
            // ════════════════════════════════════════════════
            System.out.println("=== ESTADO INICIAL ===");
            tablero.mostrarTablero();
            Thread.sleep(2000);

            // ════════════════════════════════════════════════
            // INICIAR SIMULACIÓN
            // ════════════════════════════════════════════════
            System.out.println("Iniciando simulación...");
            System.out.println("Arrancando 400 hilos...\n");
            tablero.iniciarSimulacion();

            // Esperar a que todos los hilos arranquen
            Thread.sleep(2000);

            // ════════════════════════════════════════════════
            // BUCLE DE GENERACIONES
            // ════════════════════════════════════════════════
            for (int gen = 1; gen <= numeroGeneraciones; gen++) {
                // ┌──────────────────────────────────────────────┐
                // │ PASO 1: DAR PERMISO A LAS CELDAS             │
                // └──────────────────────────────────────────────┘
                Celda.permitirAvance();

                // ┌──────────────────────────────────────────────┐
                // │ PASO 2: ESPERAR EN LA BARRERA                │
                // │ (Controlador + 400 celdas = 401 total)       │
                // └──────────────────────────────────────────────┘
                tablero.getBarreraSincronizacion().await();

                // ┌──────────────────────────────────────────────┐
                // │ PASO 3: BLOQUEAR PARA PRÓXIMA GENERACIÓN     │
                // └──────────────────────────────────────────────┘
                Celda.bloquearAvance();

                // ┌──────────────────────────────────────────────┐
                // │ PASO 4: INCREMENTAR GENERACIÓN               │
                // └──────────────────────────────────────────────┘
                tablero.siguienteGeneracion();

                // ┌──────────────────────────────────────────────┐
                // │ PASO 5: MOSTRAR TABLERO ACTUALIZADO          │
                // └──────────────────────────────────────────────┘
                tablero.mostrarTablero();

                // ┌──────────────────────────────────────────────┐
                // │ PASO 6: VERIFICAR SI ESTÁ VACÍO              │
                // └──────────────────────────────────────────────┘
                if (tablero.estaVacio()) {
                    System.out.println("\n⚠️  Todas las células han muerto.");
                    System.out.println("Fin de la simulación en generación " + gen);
                    break;
                }

                // ┌──────────────────────────────────────────────┐
                // │ PASO 7: PAUSA PARA VISUALIZACIÓN             │
                // └──────────────────────────────────────────────┘
                Thread.sleep(delayMilisegundos);
            }

            // ════════════════════════════════════════════════
            // FINALIZACIÓN
            // ════════════════════════════════════════════════
            System.out.println("\n╔════════════════════════════════════════╗");
            System.out.println("║   SIMULACIÓN FINALIZADA                ║");
            System.out.println("╠════════════════════════════════════════╣");
            System.out.println("║ Generaciones completadas: " +
                    String.format("%3d", tablero.getGeneracion()) + "          ║");
            System.out.println("╚════════════════════════════════════════╝");

            // Detener todos los hilos de celdas
            tablero.detenerSimulacion();

        } catch (InterruptedException e) {
            System.err.println("Controlador interrumpido: " + e.getMessage());
            tablero.detenerSimulacion();
        } catch (BrokenBarrierException e) {
            System.err.println("Error en barrera de sincronización: " + e.getMessage());
            e.printStackTrace();
            tablero.detenerSimulacion();
        }
    }
}