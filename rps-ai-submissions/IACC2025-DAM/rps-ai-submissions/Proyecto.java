package codigo;

import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

// Clase que representa una celda como un hilo independiente
class Celda extends Thread {
    private final int fila;
    private final int columna;
    private final Tablero tablero;
    private final CyclicBarrier barrera;
    private boolean estadoActual;
    private boolean siguienteEstado;
    private volatile boolean detener = false;

    public Celda(int fila, int columna, boolean estadoInicial, Tablero tablero, CyclicBarrier barrera) {
        this.fila = fila;
        this.columna = columna;
        this.estadoActual = estadoInicial;
        this.siguienteEstado = estadoInicial;
        this.tablero = tablero;
        this.barrera = barrera;
    }

    @Override
    public void run() {
        try {
            while (!detener) {
                // Fase 1: Calcular el siguiente estado basado en vecinos
                int vecinosVivos = tablero.contarVecinosVivos(fila, columna);
                siguienteEstado = calcularSiguienteEstado(vecinosVivos);

                // Esperar a que todas las celdas calculen su siguiente estado
                barrera.await();

                // Fase 2: Actualizar el estado actual
                estadoActual = siguienteEstado;

                // Esperar a que todas las celdas actualicen su estado
                barrera.await();
            }
        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
        }
    }

    private boolean calcularSiguienteEstado(int vecinosVivos) {
        if (estadoActual) {
            // Celda viva
            return vecinosVivos == 2 || vecinosVivos == 3; // Supervivencia
        } else {
            // Celda muerta
            return vecinosVivos == 3; // Nacimiento
        }
    }

    public void detener() {
        this.detener = true;
    }

    public boolean isViva() {
        return estadoActual;
    }

    public void setEstado(boolean estado) {
        this.estadoActual = estado;
        this.siguienteEstado = estado;
    }
}

// Clase que representa el tablero del juego
class Tablero {
    private final int filas;
    private final int columnas;
    private final Celda[][] celdas;
    private final CyclicBarrier barrera;

    public Tablero(int filas, int columnas, double probabilidadVida) {
        this.filas = filas;
        this.columnas = columnas;
        this.celdas = new Celda[filas][columnas];

        // La barrera espera a todas las celdas + 1 (el controlador)
        int totalHilos = filas * columnas + 1;
        this.barrera = new CyclicBarrier(totalHilos);

        inicializarTablero(probabilidadVida);
    }

    private void inicializarTablero(double probabilidadVida) {
        Random random = new Random();
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                boolean estadoInicial = random.nextDouble() < probabilidadVida;
                celdas[i][j] = new Celda(i, j, estadoInicial, this, barrera);
            }
        }
    }

    public void iniciarCeldas() {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                celdas[i][j].start();
            }
        }
    }

    public void detenerCeldas() {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                celdas[i][j].detener();
            }
        }
    }

    public int contarVecinosVivos(int fila, int columna) {
        int contador = 0;

        // Revisar las 8 celdas vecinas
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue; // Saltar la celda actual

                int filaVecina = fila + i;
                int columnaVecina = columna + j;

                // Verificar límites del tablero
                if (filaVecina >= 0 && filaVecina < filas &&
                        columnaVecina >= 0 && columnaVecina < columnas) {
                    if (celdas[filaVecina][columnaVecina].isViva()) {
                        contador++;
                    }
                }
            }
        }

        return contador;
    }

    public void mostrarTablero(int generacion) {
        System.out.println("\n=== Generación " + generacion + " ===");
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                System.out.print(celdas[i][j].isViva() ? "█ " : "· ");
            }
            System.out.println();
        }
    }

    public CyclicBarrier getBarrera() {
        return barrera;
    }

    public int contarCeldasVivas() {
        int contador = 0;
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                if (celdas[i][j].isViva()) {
                    contador++;
                }
            }
        }
        return contador;
    }
}

// Clase controladora que gestiona las generaciones
class ControladorJuego extends Thread {
    private final Tablero tablero;
    private final int numGeneraciones;
    private final CyclicBarrier barrera;

    public ControladorJuego(Tablero tablero, int numGeneraciones) {
        this.tablero = tablero;
        this.numGeneraciones = numGeneraciones;
        this.barrera = tablero.getBarrera();
    }

    @Override
    public void run() {
        try {
            // Mostrar estado inicial
            tablero.mostrarTablero(0);
            System.out.println("Células vivas: " + tablero.contarCeldasVivas());

            for (int generacion = 1; generacion <= numGeneraciones; generacion++) {
                // Esperar a que todas las celdas calculen su siguiente estado
                barrera.await();

                // Esperar a que todas las celdas actualicen su estado
                barrera.await();

                // Mostrar el nuevo estado del tablero
                tablero.mostrarTablero(generacion);
                System.out.println("Células vivas: " + tablero.contarCeldasVivas());

                // Pausa para visualización
                Thread.sleep(500);
            }

            // Detener todas las celdas
            tablero.detenerCeldas();

            // Hacer dos await más para liberar a las celdas que están esperando
            barrera.await();
            barrera.await();

        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
        }
    }
}

// Clase principal
public class Proyecto {
    public static void main(String[] args) {
        // Configuración del juego
        final int FILAS = 20;
        final int COLUMNAS = 40;
        final double PROBABILIDAD_VIDA = 0.3; // 30% de probabilidad de célula viva
        final int NUM_GENERACIONES = 50;

        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║        JUEGO DE LA VIDA DE CONWAY - PSP               ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.println("\nConfiguración:");
        System.out.println("• Tamaño del tablero: " + FILAS + "x" + COLUMNAS);
        System.out.println("• Número de generaciones: " + NUM_GENERACIONES);
        System.out.println("• Probabilidad inicial de vida: " + (PROBABILIDAD_VIDA * 100) + "%");

        // Crear el tablero
        Tablero tablero = new Tablero(FILAS, COLUMNAS, PROBABILIDAD_VIDA);

        // Iniciar todas las celdas (hilos)
        tablero.iniciarCeldas();

        // Crear y ejecutar el controlador
        ControladorJuego controlador = new ControladorJuego(tablero, NUM_GENERACIONES);
        controlador.start();

        try {
            // Esperar a que termine el controlador
            controlador.join();
            System.out.println("\n✓ Simulación completada con éxito");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Error en la simulación");
        }
    }
}
