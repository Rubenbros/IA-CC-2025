//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {

        Organizador o1 = new Organizador();
        Thread_Comprobar c1 = new Thread_Comprobar(o1);

        c1.start();
        o1.start();
    }
}
