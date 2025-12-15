import java.util.Scanner;

public class Organizador extends Thread {

    Scanner sc = new Scanner(System.in);
    boolean avanzar = false;
    boolean terminar = false;

    @Override
    public void run() {
        while (!terminar) {
            System.out.println("Pulsa 1 para avanzar turno (0 para salir)");
            int op = sc.nextInt();

            synchronized (this) {
                if (op == 1) {
                    avanzar = true;
                    notify(); // despierta al juego
                } else {
                    terminar = true;
                    notify();
                }
            }
        }
    }
}
