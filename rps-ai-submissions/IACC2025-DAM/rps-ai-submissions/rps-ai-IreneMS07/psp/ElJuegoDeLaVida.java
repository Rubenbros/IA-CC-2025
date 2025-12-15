//IreneMartínezSantolaria2DAMV

import java.util.Random;
import java.util.concurrent.CountDownLatch;

public class ElJuegoDeLaVida {


    private static final int TAMANO_TABLERO = 15;
    private static final int NUM_GENERACIONES = 25;
    private static final int VIVA = 1;
    private static final int MUERTA = 0;
    private static final long RETRASO_MS = 200;
    private static final int NUM_CELDAS=TAMANO_TABLERO*TAMANO_TABLERO;


    private static class Tablero {
        private final int tamano;
        private int[][] estadoActual;
        private int[][] estadoSiguiente;

        public Tablero() {
            this.tamano = TAMANO_TABLERO;
            this.estadoActual = new int[tamano][tamano];
            this.estadoSiguiente = new int[tamano][tamano];
            inicializarAleatoriamente();
        }

        private void inicializarAleatoriamente() {
            Random rand = new Random();
            for (int i = 0; i < tamano; i++) {
                for (int j = 0; j < tamano; j++) {
                    estadoActual[i][j] = rand.nextDouble() < 0.5 ? VIVA : MUERTA;
                }
            }
        }

        public int contarVecinasVivas(int fila, int columna) {
            int vecinasVivas = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (i == 0 && j == 0) continue;

                    int filaVecina = (fila + i + tamano) % tamano;
                    int columnaVecina = (columna + j + tamano) % tamano;

                    if (estadoActual[filaVecina][columnaVecina] == VIVA) {
                        vecinasVivas++;
                    }
                }
            }
            return vecinasVivas;
        }

        public synchronized void intercambiarGeneraciones() {
            int[][] temp = estadoActual;
            estadoActual = estadoSiguiente;
            estadoSiguiente = temp;
        }

        public synchronized void mostrar(int generacion) {
            System.out.println("\n--- Generación " + generacion + "---");
            for (int i = 0; i < tamano; i++) {
                for (int j = 0; j < tamano; j++) {
                    System.out.print(estadoActual[i][j] == VIVA ? "██" : ".");
                }
                System.out.println();
            }
        }
    }

    private static class Celda extends Thread {
        private final Tablero tablero;
        private final int fila;
        private final int columna;
        private int estadoSiguiente;

        private CountDownLatch cerrojoLectura;
        private CountDownLatch cerrojoActualizacion;

        public Celda(Tablero tablero, int fila, int columna) {
            this.tablero = tablero;
            this.fila = fila;
            this.columna = columna;
            this.estadoSiguiente = MUERTA;
        }

        public void configurarCerrojos(CountDownLatch cerrojoLectura, CountDownLatch cerrojoActualizacion) {
            this.cerrojoLectura = cerrojoLectura;
            this.cerrojoActualizacion = cerrojoActualizacion;
        }

        private void calcularEstadoSiguiente() {
            int vecinasVivas = tablero.contarVecinasVivas(fila, columna);
            int estadoActualCelda = tablero.estadoActual[fila][columna];

            if (estadoActualCelda == VIVA) {
                if (vecinasVivas == 2 || vecinasVivas == 3) {
                    estadoSiguiente = VIVA;
                } else {
                    estadoSiguiente = MUERTA;
                }
            } else {
                if (vecinasVivas == 3) {
                    estadoSiguiente = VIVA;
                } else {
                    estadoSiguiente = MUERTA;
                }
            }
        }
        private void actualizarEstado() {
            tablero.estadoSiguiente[fila][columna] = estadoSiguiente;
        }
        @Override
        public void run() {
            while (!isInterrupted()) {
                try {

                    cerrojoLectura.await();
                    calcularEstadoSiguiente();
                    cerrojoActualizacion.countDown();

                    cerrojoLectura.await();
                    actualizarEstado();
                    cerrojoActualizacion.countDown();

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }


    private static class ControladorJuego extends Thread {
        private final Tablero tablero;
        private final Celda[][] celdas;
        private final int numGeneraciones;
        private final int numCeldas;

        private final CountDownLatch iniciarFase1LatchInicial;
        private final CountDownLatch fase1FinalizadaLatchInicial;

        public ControladorJuego(Tablero tablero, Celda[][] celdas, int numGeneraciones, CountDownLatch iniciarFase1Latch,CountDownLatch fase1FinalizadaLatch) {
            this.tablero = tablero;
            this.celdas = celdas;
            this.numGeneraciones = numGeneraciones;
            this.numCeldas = celdas.length * celdas[0].length;
            this.iniciarFase1LatchInicial=iniciarFase1Latch;
            this.fase1FinalizadaLatchInicial=fase1FinalizadaLatch;
        }

        @Override
        public void run() {
           tablero.mostrar(0);
           for(int gen=1; gen<=numGeneraciones;gen++){
                CountDownLatch iniciarFase1Latch;
                CountDownLatch fase1FinalizadaLatch;

            if ( gen == 1) {
                iniciarFase1Latch = iniciarFase1LatchInicial;
                fase1FinalizadaLatch = fase1FinalizadaLatchInicial;
            }else {
                iniciarFase1Latch = new CountDownLatch(numCeldas);
                fase1FinalizadaLatch = new CountDownLatch(numCeldas);

                for (int i = 0; i < tablero.tamano; i++) {
                    for (int j = 0; j < tablero.tamano; j++) {
                        celdas[i][j].configurarCerrojos(iniciarFase1Latch, fase1FinalizadaLatch);
                    }
                }
            }
            try {
               for(int i =0;i<numCeldas;i++){
                   iniciarFase1Latch.countDown();
               }
               fase1FinalizadaLatchInicial.await();

               tablero.intercambiarGeneraciones();
               tablero.mostrar(gen);

               Thread.sleep(RETRASO_MS);

               CountDownLatch iniciarFase2Latch = new CountDownLatch(numCeldas);
               CountDownLatch fase2FinalizadaLatch = new CountDownLatch(numCeldas);

               for (int i = 0; i < tablero.tamano; i++) {
                    for (int j = 0; j < tablero.tamano; j++) {
                         celdas[i][j].configurarCerrojos(iniciarFase2Latch, fase2FinalizadaLatch);
                    }
               }
                    for (int i = 0; i < numCeldas; i++) {
                        iniciarFase2Latch.countDown();
                    }
                    fase2FinalizadaLatch.await();

                } catch (InterruptedException e) {
                    System.out.println("Controlador de juego interrumpido");
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            System.out.println("\nSimulación del juego de la vida finaliza. Interrumpiendo Celdas");
            for (int i = 0; i < tablero.tamano; i++) {
                for (int j = 0; j < tablero.tamano; j++) {
                    celdas[i][j].interrupt();

                }
            }
        }
    }
    public static void main(String[] args) {
        Tablero tablero = new Tablero();

        Celda[][] celdas = new Celda[TAMANO_TABLERO][TAMANO_TABLERO];
        for (int i = 0; i < TAMANO_TABLERO; i++) {
            for (int j = 0; j < TAMANO_TABLERO; j++) {
                celdas[i][j] = new Celda(tablero, i, j);
            }
        }
        CountDownLatch iniciarFase1Latch = new CountDownLatch(NUM_CELDAS);
        CountDownLatch fase1FinalizadaLatch = new CountDownLatch(NUM_CELDAS);

        for(int i=0;i<TAMANO_TABLERO;i++){
            for(int j=0;j<TAMANO_TABLERO;j++){
                celdas[i][j].configurarCerrojos(iniciarFase1Latch, fase1FinalizadaLatch);

            }
        }
        for(int i =0; i<TAMANO_TABLERO;i++){
            for(int j=0;j<TAMANO_TABLERO;j++){
                celdas[i][j].start();
            }
        }
        ControladorJuego controlador= new ControladorJuego(tablero, celdas,NUM_GENERACIONES,iniciarFase1Latch, fase1FinalizadaLatch);

        controlador.start();
        try {
            controlador.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        System.out.println("Proceso finalizado");
    }
 }













