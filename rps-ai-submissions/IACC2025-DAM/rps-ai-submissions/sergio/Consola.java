package com.sgl.conway.juegovida;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

public class Consola {

    private static final BufferedReader br =
            new BufferedReader(new InputStreamReader(System.in));

    public static int leerEntero(String mensaje, int valorPorDefecto)
            throws IOException {

        System.out.print(mensaje + " [Por defecto: " + valorPorDefecto + "]: ");
        String linea = br.readLine();

        if (linea == null || linea.trim().isEmpty()) {
            return valorPorDefecto;
        }

        return Integer.parseInt(linea.trim());
    }
}
