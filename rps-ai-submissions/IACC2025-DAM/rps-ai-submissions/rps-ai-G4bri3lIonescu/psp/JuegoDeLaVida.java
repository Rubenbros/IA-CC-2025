import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Celda del Juego de la Vida
 * Cada celda es un hilo independiente que evalúa su estado
 */
class Celda extends Thread {
    private final int fila;
    private final int columna;
    private final Tablero tablero;
    private final CyclicBarrier barrera;
    private volatile boolean viva;
    private volatile boolean proximoEstado;
    private volatile boolean ejecutando = true;
    private volatile boolean calcular = false;
    private final Lock lock = new ReentrantLock();

    public Celda(int fila, int columna, boolean estadoInicial, Tablero tablero, CyclicBarrier barrera) {
        this.fila = fila;
        this.columna = columna;
        this.viva = estadoInicial;
        this.proximoEstado = estadoInicial;
        this.tablero = tablero;
        this.barrera = barrera;
    }

    @Override
    public void run() {
        try {
            while (ejecutando) {
                // Esperar señal para calcular
                synchronized (this) {
                    while (!calcular && ejecutando) {
                        wait();
                    }
                    if (!ejecutando) break;
                    calcular = false;
                }

                // Calcular el próximo estado
                calcularProximoEstado();

                // Esperar a que todas las celdas terminen de calcular
                barrera.await();
            }
        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Calcula el próximo estado basado en las reglas del Juego de la Vida
     */
    private void calcularProximoEstado() {
        int vecinasVivas = contarVecinasVivas();

        if (viva) {
            // Celda viva: supervivencia con 2 o 3 vecinas
            proximoEstado = (vecinasVivas == 2 || vecinasVivas == 3);
        } else {
            // Celda muerta: nacimiento con exactamente 3 vecinas
            proximoEstado = (vecinasVivas == 3);
        }
    }

    /**
     * Cuenta el número de celdas vecinas vivas
     * Usa topología toroidal (los bordes se conectan)
     */
    private int contarVecinasVivas() {
        int count = 0;
        int filas = tablero.getFilas();
        int columnas = tablero.getColumnas();

        // Revisar las 8 celdas vecinas
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue; // Saltar la celda actual

                // Topología toroidal: los bordes se conectan
                int vecinaFila = (fila + i + filas) % filas;
                int vecinaColumna = (columna + j + columnas) % columnas;

                if (tablero.getCelda(vecinaFila, vecinaColumna).estaViva()) {
                    count++;
                }
            }
        }

        return count;
    }

    /**
     * Actualiza el estado de la celda con el próximo estado calculado
     */
    public void actualizarEstado() {
        viva = proximoEstado;
    }

    /**
     * Señala a la celda que debe calcular su próximo estado
     */
    public synchronized void iniciarCalculo() {
        calcular = true;
        notify();
    }

    public boolean estaViva() {
        return viva;
    }

    public void setViva(boolean viva) {
        this.viva = viva;
        this.proximoEstado = viva;
    }

    public void detener() {
        synchronized (this) {
            ejecutando = false;
            notify();
        }
    }

    public int getFila() {
        return fila;
    }

    public int getColumna() {
        return columna;
    }
}

/**
 * Tablero del Juego de la Vida
 * Contiene la matriz de celdas
 */
class Tablero {
    private final int filas;
    private final int columnas;
    private final Celda[][] celdas;

    public Tablero(int filas, int columnas) {
        this.filas = filas;
        this.columnas = columnas;
        this.celdas = new Celda[filas][columnas];
    }

    public void setCelda(int fila, int columna, Celda celda) {
        celdas[fila][columna] = celda;
    }

    public Celda getCelda(int fila, int columna) {
        return celdas[fila][columna];
    }

    public int getFilas() {
        return filas;
    }

    public int getColumnas() {
        return columnas;
    }

    /**
     * Muestra el estado actual del tablero en consola
     */
    public void mostrar() {
        System.out.println("+" + "-".repeat(columnas * 2) + "+");
        for (int i = 0; i < filas; i++) {
            System.out.print("|");
            for (int j = 0; j < columnas; j++) {
                System.out.print(celdas[i][j].estaViva() ? "█ " : "· ");
            }
            System.out.println("|");
        }
        System.out.println("+" + "-".repeat(columnas * 2) + "+");
    }

    /**
     * Muestra el tablero con coordenadas
     */
    public void mostrarConCoordenadas() {
        // Mostrar números de columna
        System.out.print("    ");
        for (int j = 0; j < columnas; j++) {
            System.out.print(j + " ");
        }
        System.out.println();

        System.out.println("  +" + "-".repeat(columnas * 2) + "+");
        for (int i = 0; i < filas; i++) {
            System.out.print(i + " |");
            for (int j = 0; j < columnas; j++) {
                System.out.print(celdas[i][j].estaViva() ? "█ " : "· ");
            }
            System.out.println("|");
        }
        System.out.println("  +" + "-".repeat(columnas * 2) + "+");
    }

    /**
     * Cuenta el número total de celdas vivas
     */
    public int contarCeldasVivas() {
        int count = 0;
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                if (celdas[i][j].estaViva()) {
                    count++;
                }
            }
        }
        return count;
    }
}

/**
 * Controlador del Juego de la Vida
 * Gestiona las generaciones y la sincronización
 */
class ControladorJuego {
    private final Tablero tablero;
    private final CyclicBarrier barrera;
    private int generacion = 0;
    private static final int MAX_HISTORIAL = 10; // Detectar ciclos de hasta 10 generaciones
    private final java.util.List<String> historialEstados = new java.util.ArrayList<>();

    public ControladorJuego(int filas, int columnas, boolean[][] estadoInicial) {
        this.tablero = new Tablero(filas, columnas);

        // Barrera para sincronizar el cálculo de todas las celdas
        this.barrera = new CyclicBarrier(filas * columnas, () -> {
            // Cuando todas las celdas terminan de calcular, actualizar sus estados
            actualizarTodasLasCeldas();
        });

        inicializarTablero(estadoInicial);
    }

    public ControladorJuego(int filas, int columnas, double densidadInicial) {
        this.tablero = new Tablero(filas, columnas);

        // Barrera para sincronizar el cálculo de todas las celdas
        this.barrera = new CyclicBarrier(filas * columnas, () -> {
            // Cuando todas las celdas terminan de calcular, actualizar sus estados
            actualizarTodasLasCeldas();
        });

        inicializarTableroAleatorio(densidadInicial);
    }

    /**
     * Inicializa el tablero con un estado específico
     */
    private void inicializarTablero(boolean[][] estadoInicial) {
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                boolean estadoCelda = (estadoInicial != null && i < estadoInicial.length &&
                        j < estadoInicial[i].length) ? estadoInicial[i][j] : false;
                Celda celda = new Celda(i, j, estadoCelda, tablero, barrera);
                tablero.setCelda(i, j, celda);
            }
        }
    }

    /**
     * Inicializa el tablero con un patrón aleatorio
     */
    private void inicializarTableroAleatorio(double densidad) {
        Random random = new Random();

        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                boolean estadoInicial = random.nextDouble() < densidad;
                Celda celda = new Celda(i, j, estadoInicial, tablero, barrera);
                tablero.setCelda(i, j, celda);
            }
        }
    }

    /**
     * Actualiza el estado de todas las celdas
     */
    private void actualizarTodasLasCeldas() {
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                tablero.getCelda(i, j).actualizarEstado();
            }
        }
    }

    /**
     * Inicia la simulación
     */
    public void iniciar() {
        System.out.println("=== JUEGO DE LA VIDA ===");
        System.out.println("Tablero: " + tablero.getFilas() + "x" + tablero.getColumnas());
        System.out.println();

        // Iniciar todos los hilos de las celdas
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                tablero.getCelda(i, j).start();
            }
        }
    }

    /**
     * Avanza una generación
     */
    public boolean avanzarGeneracion() {
        // Señalar a todas las celdas que calculen su próximo estado
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                tablero.getCelda(i, j).iniciarCalculo();
            }
        }

        // Esperar un momento para que todas las celdas procesen
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        generacion++;

        // Obtener estado actual del tablero
        String estadoActual = obtenerEstadoTablero();

        // Verificar si el estado ya existió antes (ciclo o estabilidad)
        if (historialEstados.contains(estadoActual)) {
            int indiceCiclo = historialEstados.indexOf(estadoActual);
            int longitudCiclo = historialEstados.size() - indiceCiclo;

            System.out.println("\n>>> Generación " + generacion + " <<<");
            System.out.println("Celdas vivas: " + tablero.contarCeldasVivas());
            tablero.mostrar();

            if (longitudCiclo == 1) {
                System.out.println("\n*** PATRÓN ESTABLE DETECTADO ***");
                System.out.println("El tablero ha alcanzado un estado estático.");
            } else {
                System.out.println("\n*** CICLO DETECTADO ***");
                System.out.println("El patrón se repite cada " + longitudCiclo + " generación(es).");
            }
            return false; // Indicar que se debe detener
        }

        // Verificar si todas las celdas están muertas
        if (tablero.contarCeldasVivas() == 0) {
            System.out.println("\n>>> Generación " + generacion + " <<<");
            System.out.println("Celdas vivas: 0");
            tablero.mostrar();
            System.out.println("\n*** EXTINCIÓN TOTAL ***");
            System.out.println("Todas las celdas han muerto.");
            return false;
        }

        // Agregar estado al historial
        historialEstados.add(estadoActual);

        // Mantener solo los últimos MAX_HISTORIAL estados
        if (historialEstados.size() > MAX_HISTORIAL) {
            historialEstados.remove(0);
        }

        System.out.println("\n>>> Generación " + generacion + " <<<");
        System.out.println("Celdas vivas: " + tablero.contarCeldasVivas());
        tablero.mostrar();

        try {
            Thread.sleep(400); // Pausa entre generaciones
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return true; // Continuar
    }

    /**
     * Obtiene una representación en String del estado actual del tablero
     */
    private String obtenerEstadoTablero() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                sb.append(tablero.getCelda(i, j).estaViva() ? '1' : '0');
            }
        }
        return sb.toString();
    }

    /**
     * Detiene la simulación
     */
    public void detener() {
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                tablero.getCelda(i, j).detener();
            }
        }

        // Esperar a que todos los hilos terminen
        for (int i = 0; i < tablero.getFilas(); i++) {
            for (int j = 0; j < tablero.getColumnas(); j++) {
                try {
                    tablero.getCelda(i, j).join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    public Tablero getTablero() {
        return tablero;
    }

    public int getGeneracion() {
        return generacion;
    }
}

/**
 * Clase principal
 */
public class JuegoDeLaVida {

    /**
     * Permite al usuario seleccionar manualmente las celdas vivas
     */

    /**
    * NOTA: Si se intenta colocar celdas de forma que generen un bucle infinito o no haya celdas suficientes para continuar/iniciar el juego,
     * el programa automáticamente detectará esto y finalizará su ejecución.
    */
    private static boolean[][] seleccionManual(int filas, int columnas) {
        Scanner scanner = new Scanner(System.in);
        boolean[][] estado = new boolean[filas][columnas];

        System.out.println("\n=== SELECCIÓN MANUAL DE CELDAS VIVAS ===");
        System.out.println("Introduce las coordenadas de las celdas vivas.");
        System.out.println("Formato: fila columna (ejemplo: 2 3)");
        System.out.println("Escribe 'fin' cuando termines.\n");

        // Mostrar tablero vacío con coordenadas
        Tablero tableroTemp = new Tablero(filas, columnas);
        CyclicBarrier barrierTemp = new CyclicBarrier(1);
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                tableroTemp.setCelda(i, j, new Celda(i, j, false, tableroTemp, barrierTemp));
            }
        }
        tableroTemp.mostrarConCoordenadas();

        while (true) {
            System.out.print("\nCoordenadas (o 'fin'). Ejemplo: 4 6 (fila 4, columna 6)");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("fin")) {
                break;
            }

            try {
                String[] partes = input.split("\\s+");
                if (partes.length != 2) {
                    System.out.println("Error: Debes introducir dos números (fila y columna)");
                    continue;
                }

                int fila = Integer.parseInt(partes[0]);
                int columna = Integer.parseInt(partes[1]);
                /**
                 * Manejo de errores en caso de que se seleccionen celdas fuera del rango del tablero
                 */
                if (fila < 0 || fila >= filas || columna < 0 || columna >= columnas) {
                    System.out.println("Error: Coordenadas fuera de rango. Fila: 0-" + (filas-1) + ", Columna: 0-" + (columnas-1));
                    continue;
                }

                if (estado[fila][columna]) {
                    System.out.println("Celda (" + fila + "," + columna + ") ya está viva. Se desmarca.");
                    estado[fila][columna] = false;
                } else {
                    System.out.println("Celda (" + fila + "," + columna + ") marcada como viva.");
                    estado[fila][columna] = true;
                }

                // Actualizar visualización
                for (int i = 0; i < filas; i++) {
                    for (int j = 0; j < columnas; j++) {
                        tableroTemp.getCelda(i, j).setViva(estado[i][j]);
                    }
                }
                tableroTemp.mostrarConCoordenadas();

            } catch (NumberFormatException e) {
                System.out.println("Error: Entrada inválida. Usa números enteros.");
            }
        }

        return estado;
    }

    /**
     * Permite usar patrones predefinidos
     */
    private static boolean[][] obtenerPatronPredefinido(String patron, int filas, int columnas) {
        boolean[][] estado = new boolean[filas][columnas];

        switch (patron.toLowerCase()) {
            case "glider":
                // Patrón Glider (planeador)
                if (filas >= 5 && columnas >= 5) {
                    estado[1][2] = true;
                    estado[2][3] = true;
                    estado[3][1] = true;
                    estado[3][2] = true;
                    estado[3][3] = true;
                }
                break;

            case "blinker":
                // Patrón Blinker (oscilador)
                if (filas >= 5 && columnas >= 5) {
                    estado[2][1] = true;
                    estado[2][2] = true;
                    estado[2][3] = true;
                }
                break;

            case "toad":
                // Patrón Toad (oscilador)
                if (filas >= 6 && columnas >= 6) {
                    estado[2][2] = true;
                    estado[2][3] = true;
                    estado[2][4] = true;
                    estado[3][1] = true;
                    estado[3][2] = true;
                    estado[3][3] = true;
                }
                break;

            case "beacon":
                // Patrón Beacon (oscilador)
                if (filas >= 6 && columnas >= 6) {
                    estado[1][1] = true;
                    estado[1][2] = true;
                    estado[2][1] = true;
                    estado[3][4] = true;
                    estado[4][3] = true;
                    estado[4][4] = true;
                }
                break;
        }

        return estado;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Configuración del tablero
        System.out.println("=== CONFIGURACIÓN DEL JUEGO DE LA VIDA ===");
        System.out.print("Número de filas (default 10): ");
        String inputFilas = scanner.nextLine().trim();
        int FILAS = inputFilas.isEmpty() ? 10 : Integer.parseInt(inputFilas);

        System.out.print("Número de columnas (default 10): ");
        String inputColumnas = scanner.nextLine().trim();
        int COLUMNAS = inputColumnas.isEmpty() ? 10 : Integer.parseInt(inputColumnas);

        System.out.print("Número de generaciones (default 50): ");
        String inputGen = scanner.nextLine().trim();
        int NUM_GENERACIONES = inputGen.isEmpty() ? 50 : Integer.parseInt(inputGen);

        // Selección del modo de inicialización
        System.out.println("\n=== MODO DE INICIALIZACIÓN ===");
        System.out.println("1. Manual (seleccionar celdas)");
        System.out.println("2. Aleatorio");
        System.out.println("3. Patrón predefinido");
        System.out.print("Selecciona una opción (1-3): ");
        String opcion = scanner.nextLine().trim();

        boolean[][] estadoInicial = null;
        ControladorJuego juego = null;

        switch (opcion) {
            case "1":
                // Modo manual
                estadoInicial = seleccionManual(FILAS, COLUMNAS);
                juego = new ControladorJuego(FILAS, COLUMNAS, estadoInicial);
                break;

            case "2":
                // Modo aleatorio
                System.out.print("Densidad inicial (0.0 - 1.0, default 0.3): ");
                String inputDens = scanner.nextLine().trim();
                double DENSIDAD = inputDens.isEmpty() ? 0.3 : Double.parseDouble(inputDens);
                juego = new ControladorJuego(FILAS, COLUMNAS, DENSIDAD);
                break;

            case "3":
                // Patrones predefinidos
                System.out.println("\nPatrones disponibles:");
                System.out.println("- glider: Planeador que se mueve");
                System.out.println("- blinker: Oscilador simple");
                System.out.println("- toad: Oscilador tipo sapo");
                System.out.println("- beacon: Oscilador tipo faro");
                System.out.print("Selecciona un patrón: ");
                String patron = scanner.nextLine().trim();
                estadoInicial = obtenerPatronPredefinido(patron, FILAS, COLUMNAS);
                juego = new ControladorJuego(FILAS, COLUMNAS, estadoInicial);
                break;

            default:
                System.out.println("Opción inválida. Usando modo aleatorio por defecto.");
                juego = new ControladorJuego(FILAS, COLUMNAS, 0.3);
        }

        // Mostrar estado inicial
        System.out.println("\n>>> Estado Inicial <<<");
        System.out.println("Celdas vivas: " + juego.getTablero().contarCeldasVivas());
        juego.getTablero().mostrar();

        // Iniciar simulación
        juego.iniciar();

        // Ejecutar generaciones
        for (int i = 0; i < NUM_GENERACIONES; i++) {
            boolean continuar = juego.avanzarGeneracion();
            if (!continuar) {
                System.out.println("\nSimulación detenida en la generación " + juego.getGeneracion());
                break;
            }
        }

        // Detener simulación
        juego.detener();

        System.out.println("\n=== SIMULACIÓN FINALIZADA ===");
        System.out.println("Generaciones ejecutadas: " + juego.getGeneracion());

        scanner.close();
    }
}