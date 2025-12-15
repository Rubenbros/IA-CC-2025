import java.util.Scanner;

public class Thread_Comprobar extends Thread {

    private Organizador organizador;
    private Scanner sc = new Scanner(System.in);

    private int filas = 15;
    private int columnas = 15;
    boolean[][] mundo = new boolean[filas][columnas];

    public Thread_Comprobar(Organizador organizador) {
        this.organizador = organizador;
        inicializarCentro();
        colocarExtra();
    }

   
    void inicializarCentro() {
        mundo[filas / 2][columnas / 2] = true;
    }

    void colocarExtra() {
        System.out.println("¿Quieres añadir más células? (1 = sí, 0 = no)");
        int op = sc.nextInt();

        while (op == 1) {
            System.out.println("Fila:");
            int x = sc.nextInt();
            System.out.println("Columna:");
            int y = sc.nextInt();

            mundo[x][y] = true;
            imprimirMundo(mundo);

            System.out.println("¿Añadir otra? (1 = sí, 0 = no)");
            op = sc.nextInt();
        }
    }

    @Override
    public void run() {
        while (!organizador.terminar) {

            synchronized (organizador) {
                while (!organizador.avanzar && !organizador.terminar) {
                    try {
                        organizador.wait(); // espera al 1
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                organizador.avanzar = false;
            }

            if (!organizador.terminar) {
                imprimirMundo(mundo);
                mundo = generarSiguiente(mundo);
            }
        }
    }

    public int contarVecinos(boolean[][] mundo, int x, int y) {
        int v = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int nx = x + i;
                int ny = y + j;
                if (nx >= 0 && nx < mundo.length && ny >= 0 && ny < mundo[0].length) {
                    if (mundo[nx][ny]) v++;
                }
            }
        }
        return v;
    }

    public boolean[][] generarSiguiente(boolean[][] mundo) {
        boolean[][] s = new boolean[mundo.length][mundo[0].length];

        for (int x = 0; x < mundo.length; x++) {
            for (int y = 0; y < mundo[0].length; y++) {
                int v = contarVecinos(mundo, x, y);

                if (mundo[x][y]) {
                    s[x][y] = (v == 2 || v == 3);
                } else {
                    s[x][y] = (v == 3);
                }
            }
        }
        return s;
    }

    void imprimirMundo(boolean[][] mundo) {
        System.out.println("\n----------------");
        for (int i = 0; i < mundo.length; i++) {
            for (int j = 0; j < mundo[i].length; j++) {
                System.out.print(mundo[i][j] ? "█" : ".");
            }
            System.out.println();
        }
    }
}
