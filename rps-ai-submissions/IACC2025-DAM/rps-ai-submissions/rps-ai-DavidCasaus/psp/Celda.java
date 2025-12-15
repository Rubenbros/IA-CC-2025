import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class Celda extends Thread {
    private final int fila;
    private final int columna;
    private final Tablero tablero;
    private final CyclicBarrier barrera;
    private boolean estadoActual;
    private boolean siguienteEstado;
    private volatile boolean ejecutando;

    public Celda(int fila, int columna, boolean estadoInicial, Tablero tablero, CyclicBarrier barrera) {
        this.fila = fila;
        this.columna = columna;
        this.estadoActual = estadoInicial;
        this.siguienteEstado = estadoInicial;
        this.tablero = tablero;
        this.barrera = barrera;
        this.ejecutando = true;
    }

    @Override
    public void run() {
        try {
            while (ejecutando) {
                // Fase 1: Calcular siguiente estado basado en vecinos
                calcularSiguienteEstado();

                // Esperar a que todas las celdas calculen su siguiente estado
                barrera.await();

                // Fase 2: Actualizar estado actual
                estadoActual = siguienteEstado;

                // Esperar a que todas las celdas actualicen su estado
                barrera.await();
            }
        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Calcula el siguiente estado de la celda según las reglas del Juego de la Vida
     */
    private void calcularSiguienteEstado() {
        int vecinosVivos = contarVecinosVivos();

        if (estadoActual) {
            // Celda viva
            if (vecinosVivos < 2) {
                siguienteEstado = false; // Muerte por soledad
            } else if (vecinosVivos == 2 || vecinosVivos == 3) {
                siguienteEstado = true; // Supervivencia
            } else {
                siguienteEstado = false; // Muerte por sobrepoblación
            }
        } else {
            // Celda muerta
            if (vecinosVivos == 3) {
                siguienteEstado = true; // Nacimiento
            } else {
                siguienteEstado = false; // Permanece muerta
            }
        }
    }

    /**
     * Cuenta el número de vecinos vivos alrededor de esta celda
     */
    private int contarVecinosVivos() {
        int count = 0;
        int filas = tablero.getFilas();
        int columnas = tablero.getColumnas();

        // Revisar las 8 celdas vecinas
        for (int df = -1; df <= 1; df++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (df == 0 && dc == 0) continue; // Saltar la celda actual

                int vecinoFila = fila + df;
                int vecinoColumna = columna + dc;

                // Verificar límites del tablero
                if (vecinoFila >= 0 && vecinoFila < filas &&
                        vecinoColumna >= 0 && vecinoColumna < columnas) {
                    if (tablero.obtenerEstadoCelda(vecinoFila, vecinoColumna)) {
                        count++;
                    }
                }
            }
        }

        return count;
    }

    public boolean estaViva() {
        return estadoActual;
    }

    public void detener() {
        ejecutando = false;
    }

    public int getFila() {
        return fila;
    }

    public int getColumna() {
        return columna;
    }
}