package com.sgl.conway.juegovida;

import java.util.concurrent.CyclicBarrier;

public class Celda implements Runnable {

    private final int fila;
    private final int columna;
    private boolean viva;
    private boolean siguienteEstado;

    private final Tablero tablero;
    private final CyclicBarrier barrera;

    public Celda(int fila, int columna, boolean viva, Tablero tablero, CyclicBarrier barrera) {
        this.fila = fila;
        this.columna = columna;
        this.viva = viva;
        this.tablero = tablero;
        this.barrera = barrera;
    }

    @Override
    public void run() {
        try {
            while (Controlador.ejecutar) {
                calcularSiguienteEstado();
                barrera.await();
                actualizarEstado();
                barrera.await();
            }
        } catch (Exception e) {
            Thread.currentThread().interrupt();
        }
    }


    private void calcularSiguienteEstado() {
        int vecinasVivas = tablero.contarVecinasVivas(fila, columna);

        if (viva) {
            siguienteEstado = (vecinasVivas == 2 || vecinasVivas == 3);
        } else {
            siguienteEstado = (vecinasVivas == 3);
        }
    }

    private void actualizarEstado() {
        viva = siguienteEstado;
    }

    public boolean estaViva() {
        return viva;
    }
}
