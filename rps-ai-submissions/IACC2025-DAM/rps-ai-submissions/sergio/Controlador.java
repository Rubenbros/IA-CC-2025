package com.sgl.conway.juegovida;

public class Controlador implements Runnable {

    private final Tablero tablero;
    private final int maxGeneraciones;
    private int generacion = 0;

    public static volatile boolean ejecutar = true;
    public static volatile int generacionesEjecutadas = 0;

    public Controlador(Tablero tablero, int maxGeneraciones) {
        this.tablero = tablero;
        this.maxGeneraciones = maxGeneraciones;
    }

    @Override
    public void run() {
        int poblacion = tablero.contarVivas();

        if (poblacion == 0) {
            ejecutar = false;
            generacionesEjecutadas = generacion;
            System.out.println("La población se ha extinguido.");
            return;
        }

        if (generacion >= maxGeneraciones) {
            ejecutar = false;
            generacionesEjecutadas = generacion;
            return;
        }

        System.out.println("Generación " + generacion + " | Población: " + poblacion);
        tablero.mostrar();
        generacion++;

        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
