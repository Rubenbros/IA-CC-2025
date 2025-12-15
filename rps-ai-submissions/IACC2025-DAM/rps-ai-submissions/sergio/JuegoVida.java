package com.sgl.conway.juegovida;

import java.util.concurrent.CyclicBarrier;

public class JuegoVida {

    public static void main(String[] args) {

        try {
            // Configuración por defecto
            int filas = 10;
            int columnas = 10;
            int generaciones = 25;
            int habitantes = 50;

            filas = Consola.leerEntero("Filas del tablero ", filas);
            columnas = Consola.leerEntero("Columnas del tablero ", columnas);
            generaciones = Consola.leerEntero("Número de generaciones ", generaciones);
            habitantes = Consola.leerEntero("Habitantes iniciales ", habitantes);

            Tablero tablero = new Tablero(filas, columnas);

            CyclicBarrier barrera = new CyclicBarrier(
                    filas * columnas,
                    new Controlador(tablero, generaciones)
            );

            tablero.inicializar(barrera, habitantes);

            for (Celda[] fila : tablero.getCeldas()) {
                for (Celda celda : fila) {
                    new Thread(celda).start();
                }
            }

            while (Controlador.ejecutar) {
                Thread.sleep(100);
            }

            System.out.println("\n--- FIN DE LA SIMULACIÓN ---");
            System.out.println("Generaciones ejecutadas: " +
                    Controlador.generacionesEjecutadas);
            System.out.println("Población final: " +
                    tablero.contarVivas());

        } catch (Exception e) {
            System.err.println("Error en la entrada de datos.");
        }

    }
}
