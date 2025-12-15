
import java.util.concurrent.CyclicBarrier;

public class GameOfLifeMain {

    private static final int ROWS = 10;
    private static final int COLS = 10;
    private static final int GENERATIONS = 100;

    public static void main(String[] args) {

        GameState state = new GameState(ROWS, COLS);
        state.randomInit();

        System.out.println("Generación 0:");
        GameState.printBoard(state.board);

        int numCells = ROWS * COLS;

        GameController controller = new GameController(state);

        CyclicBarrier barrier = new CyclicBarrier(numCells, controller);

        Thread[][] cellThreads = new Thread[ROWS][COLS];

        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                CellWorker worker = new CellWorker(state, barrier, r, c, GENERATIONS);
                Thread t = new Thread(worker, "Cell-" + r + "-" + c);
                cellThreads[r][c] = t;
                t.start();
            }
        }

        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                try {
                    cellThreads[r][c].join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        System.out.println("Simulación finalizada.");
    }
}
