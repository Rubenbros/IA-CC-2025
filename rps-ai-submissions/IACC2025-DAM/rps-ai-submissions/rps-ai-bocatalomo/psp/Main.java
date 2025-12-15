/**
 * Clase principal para ejecutar el Juego de la Vida
 */
public class Main {
    public static void main(String[] args) {
        // ════════════════════════════════════════════════════════════
        // CONFIGURACIÓN
        // ════════════════════════════════════════════════════════════
        final int TAMAÑO = 20;                    // Tablero 20x20
        final int GENERACIONES = 50;              // Número de generaciones
        final int DELAY_MS = 500;                 // Medio segundo entre generaciones
        final double DENSIDAD_INICIAL = 0.25;     // 25% de celdas vivas
        final String PATRON_INICIAL = "aleatorio"; // o "blinker", "glider", etc.

        // ════════════════════════════════════════════════════════════
        // BIENVENIDA
        // ════════════════════════════════════════════════════════════
        System.out.println("╔════════════════════════════════════════════════╗");
        System.out.println("║                                                ║");
        System.out.println("║        JUEGO DE LA VIDA - Conway               ║");
        System.out.println("║        Programación de Procesos y Servicios    ║");
        System.out.println("║                                                ║");
        System.out.println("╚════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("⚙️  Configuración:");
        System.out.println("   • Tamaño del tablero: " + TAMAÑO + "x" + TAMAÑO);
        System.out.println("   • Total de células: " + (TAMAÑO * TAMAÑO));
        System.out.println("   • Generaciones: " + GENERACIONES);
        System.out.println("   • Delay: " + DELAY_MS + "ms");
        System.out.println("   • Patrón: " + PATRON_INICIAL);
        System.out.println();

        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // ════════════════════════════════════════════════════════════
        // CREAR TABLERO Y CONTROLADOR
        // ════════════════════════════════════════════════════════════
        Tablero tablero = new Tablero();

        // Inicializar con patrón
        if (PATRON_INICIAL.equalsIgnoreCase("aleatorio")) {
            tablero.inicializarAleatorio(DENSIDAD_INICIAL);
        } else {
            tablero.inicializarConPatron(PATRON_INICIAL);
        }

        // Crear controlador (sin pasar barreras, están dentro del tablero)
        Controlador controlador = new Controlador(tablero, GENERACIONES, DELAY_MS);

        // ════════════════════════════════════════════════════════════
        // EJECUTAR SIMULACIÓN
        // ════════════════════════════════════════════════════════════
        controlador.start();

        try {
            controlador.join(); // Esperar a que termine
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // ════════════════════════════════════════════════════════════
        // DESPEDIDA
        // ════════════════════════════════════════════════════════════
        System.out.println("\n¡Gracias por usar el Juego de la Vida!");
    }
}