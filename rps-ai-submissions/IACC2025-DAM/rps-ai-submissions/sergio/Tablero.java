package com.sgl.conway.juegovida;

import java.util.Random;
import java.util.concurrent.CyclicBarrier;

public class Tablero {

    private final int filas;
    private final int columnas;
    private final Celda[][] celdas;

    public Tablero(int filas, int columnas) {
        this.filas = filas;
        this.columnas = columnas;
        this.celdas = new Celda[filas][columnas];
    }

    public void inicializar(CyclicBarrier barrera, int habitantesIniciales) {

        Random rnd = new Random();

        boolean[][] vivos = new boolean[filas][columnas];
        int colocados = 0;

        while (colocados < habitantesIniciales) {
            int i = rnd.nextInt(filas);
            int j = rnd.nextInt(columnas);
            if (!vivos[i][j]) {
                vivos[i][j] = true;
                colocados++;
            }
        }

        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                celdas[i][j] = new Celda(i, j, vivos[i][j], this, barrera);
            }
        }
    }


    public int contarVecinasVivas(int fila, int columna) {
        int contador = 0;

        for (int i = fila - 1; i <= fila + 1; i++) {
            for (int j = columna - 1; j <= columna + 1; j++) {
                if (i == fila && j == columna) continue;

                if (i >= 0 && i < filas && j >= 0 && j < columnas) {
                    if (celdas[i][j].estaViva()) {
                        contador++;
                    }
                }
            }
        }
        return contador;
    }

    public void mostrar() {
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                System.out.print(celdas[i][j].estaViva() ? "■ " : "· ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public Celda[][] getCeldas() {
        return celdas;
    }

    public int contarVivas() {
        int total = 0;
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                if (celdas[i][j].estaViva()) {
                    total++;
                }
            }
        }
        return total;
    }

}
