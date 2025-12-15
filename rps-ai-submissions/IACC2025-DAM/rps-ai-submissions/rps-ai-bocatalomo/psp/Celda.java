import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Representa una celda del Juego de la Vida como un hilo independiente.
 */
public class Celda extends Thread {
    // Posición de la celda en el tablero
    private int fila;
    private int columna;

    // Estados de la celda (volatile para visibilidad entre hilos)
    private volatile boolean estadoActual;
    private volatile boolean estadoSiguiente;

    // Referencia al tablero (para consultar vecinos)
    private Tablero tablero;

    // Barreras de sincronización
    private CyclicBarrier barreraCalculo;
    private CyclicBarrier barreraActualizacion;
    private CyclicBarrier barreraSincronizacion;  // Nueva: Con Controlador

    // Control de ejecución
    private volatile boolean activo = true;

    // Control de avance (compartido por TODAS las celdas)
    private static final Lock lock = new ReentrantLock();
    private static final Condition avanzar = lock.newCondition();
    private static volatile boolean puedeAvanzar = false;

    /**
     * Constructor de la Celda
     */
    public Celda(int fila, int columna, Tablero tablero,
                 CyclicBarrier barreraCalculo,
                 CyclicBarrier barreraActualizacion,
                 CyclicBarrier barreraSincronizacion) {
        this.fila = fila;
        this.columna = columna;
        this.tablero = tablero;
        this.barreraCalculo = barreraCalculo;
        this.barreraActualizacion = barreraActualizacion;
        this.barreraSincronizacion = barreraSincronizacion;
        this.estadoActual = false;
        this.estadoSiguiente = false;
    }

    /**
     * Método principal del hilo
     */
    @Override
    public void run() {
        try {
            while (activo) {
                // ════════════════════════════════════════════════
                // PASO 0: ESPERAR PERMISO DEL CONTROLADOR
                // ════════════════════════════════════════════════
                lock.lock();
                try {
                    while (!puedeAvanzar && activo) {
                        avanzar.await();
                    }
                } finally {
                    lock.unlock();
                }

                if (!activo) break;

                // ════════════════════════════════════════════════
                // PASO 1: CALCULAR SIGUIENTE ESTADO
                // ════════════════════════════════════════════════
                calcularSiguienteEstado();

                // ════════════════════════════════════════════════
                // PASO 2: BARRERA - Todas las celdas calcularon
                // ════════════════════════════════════════════════
                barreraCalculo.await();

                // ════════════════════════════════════════════════
                // PASO 3: ACTUALIZAR ESTADO
                // ════════════════════════════════════════════════
                actualizarEstado();

                // ════════════════════════════════════════════════
                // PASO 4: BARRERA - Todas las celdas actualizaron
                // ════════════════════════════════════════════════
                barreraActualizacion.await();

                // ════════════════════════════════════════════════
                // PASO 5: BARRERA - Sincronizar con Controlador
                // ════════════════════════════════════════════════
                barreraSincronizacion.await();

                // Volver al inicio: esperar nueva señal
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (BrokenBarrierException e) {
            System.err.println("Barrera rota en celda [" + fila + "][" + columna + "]");
        }
    }

    /**
     * Cuenta cuántos vecinos vivos tiene esta celda
     */
    private int contarVecinosVivos() {
        int contador = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                int vecinoFila = fila + i;
                int vecinoColumna = columna + j;

                if (tablero.dentroDelTablero(vecinoFila, vecinoColumna)) {
                    // estaCeldaViva() ahora está sincronizado en Tablero
                    if (tablero.estaCeldaViva(vecinoFila, vecinoColumna)) {
                        contador++;
                    }
                }
            }
        }

        return contador;
    }

    /**
     * Aplica las reglas del Juego de la Vida
     */
    private void calcularSiguienteEstado() {
        int vecinosVivos = contarVecinosVivos();

        if (estadoActual) {
            // CELDA VIVA
            if (vecinosVivos < 2) {
                estadoSiguiente = false;  // Muerte por soledad
            } else if (vecinosVivos == 2 || vecinosVivos == 3) {
                estadoSiguiente = true;   // Supervivencia
            } else {
                estadoSiguiente = false;  // Muerte por sobrepoblación
            }
        } else {
            // CELDA MUERTA
            if (vecinosVivos == 3) {
                estadoSiguiente = true;   // Nacimiento
            } else {
                estadoSiguiente = false;  // Sigue muerta
            }
        }
    }

    /**
     * Actualiza el estado actual con el calculado
     */
    private void actualizarEstado() {
        estadoActual = estadoSiguiente;
    }

    /**
     * Devuelve si la celda está viva
     */
    public synchronized boolean estaViva() {
        return estadoActual;
    }

    /**
     * Establece el estado de la celda
     */
    public synchronized void setEstado(boolean estado) {
        this.estadoActual = estado;
        this.estadoSiguiente = estado;
    }

    /**
     * Detiene la ejecución del hilo
     */
    public void detener() {
        this.activo = false;
        lock.lock();
        try {
            avanzar.signalAll();
        } finally {
            lock.unlock();
        }
    }

    // ════════════════════════════════════════════════════════════
    // MÉTODOS STATIC PARA CONTROL DEL CONTROLADOR
    // ════════════════════════════════════════════════════════════

    /**
     * Permite que las celdas avancen una generación
     */
    public static void permitirAvance() {
        lock.lock();
        try {
            puedeAvanzar = true;
            avanzar.signalAll();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Bloquea las celdas para que no avancen sin permiso
     */
    public static void bloquearAvance() {
        lock.lock();
        try {
            puedeAvanzar = false;
        } finally {
            lock.unlock();
        }
    }
}