import java.util.InputMismatchException;
import java.util.Scanner;

public class JuegoDeLaVida {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Tablero tablero = new Tablero();

        System.out.println("🌱 ¡Bienvenido al Juego de la Vida (10x10)! 🌱");
        System.out.println("--- Configuración Inicial ---");

        // 1. Pedir posiciones iniciales al usuario
        iniciarVida(scanner, tablero);

        // 2. Simulación
        simular(scanner, tablero);

        scanner.close();
    }

    /**
     * Permite al usuario seleccionar las posiciones iniciales de las células vivas.
     */
    private static void iniciarVida(Scanner scanner, Tablero tablero) {
        System.out.println("Introduce las coordenadas (Fila Columna) para crear vida (0-9).");
        System.out.println("Escribe 'fin' cuando hayas terminado.");

        while (true) {
            System.out.print("Coordenadas (Ej: 5 3) o 'fin': ");
            String entrada = scanner.next();

            if (entrada.equalsIgnoreCase("fin")) {
                break;
            }

            try {
                int fila = Integer.parseInt(entrada);
                int columna = scanner.nextInt();

                if (fila >= 0 && fila < 10 && columna >= 0 && columna < 10) {
                    tablero.establecerVida(fila, columna);
                    System.out.println("Vida creada en (" + fila + ", " + columna + ")");
                } else {
                    System.out.println("⚠️ Coordenadas fuera del rango (0-9). Inténtalo de nuevo.");
                }

            } catch (NumberFormatException e) {
                System.out.println("⚠️ Entrada no válida. Usa 'Fila Columna' o 'fin'.");
            } catch (InputMismatchException e) {
                System.out.println("⚠️ Entrada incompleta. Asegúrate de introducir Fila Y Columna.");
                scanner.next(); // Limpiar el buffer si la segunda entrada es incorrecta
            }
        }
    }

    /**
     * Ejecuta el ciclo de simulación generación tras generación.
     */
    private static void simular(Scanner scanner, Tablero tablero) {
        int generacion = 0;

        while (true) {
            System.out.println("\n*** GENERACIÓN " + generacion + " ***");
            tablero.imprimirTablero();

            System.out.print("Pulsa **Enter** para la siguiente generación, o escribe **'salir'** para terminar: ");
            String comando = scanner.nextLine(); // Consumir línea pendiente y esperar comando

            if (comando.equalsIgnoreCase("salir")) {
                System.out.println("Simulación terminada. ¡Gracias por jugar!");
                break;
            }

            // Pasar a la siguiente generación
            tablero.siguienteGeneracion();
            generacion++;
        }
    }
}