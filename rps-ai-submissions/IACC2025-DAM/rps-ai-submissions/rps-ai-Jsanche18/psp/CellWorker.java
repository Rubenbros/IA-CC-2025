import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class CellWorker implements Runnable {

    private final GameState state;
    private final CyclicBarrier barrier;
    private final int row;
    private final int col;
    private final int maxGenerations;

    public CellWorker(GameState state, CyclicBarrier barrier,
                      int row, int col, int maxGenerations) {
        this.state = state;
        this.barrier = barrier;
        this.row = row;
        this.col = col;
        this.maxGenerations = maxGenerations;
    }

    @Override
    public void run() {
        try {
            for (int g = 0; g < maxGenerations; g++) {

                boolean nextValue = computeNextState();

                state.nextBoard[row][col] = nextValue;

                barrier.await();
            }
        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
        }
    }

    private boolean computeNextState() {
        boolean alive = state.board[row][col];
        int neighbors = countAliveNeighbors();

        if (alive) {
            if (neighbors < 2) return false;
            if (neighbors == 2 || neighbors == 3) return true;
            return false;
        } else {
            return neighbors == 3;
        }
    }

    private int countAliveNeighbors() {
        int count = 0;

        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {

                if (dr == 0 && dc == 0) continue;

                int nr = row + dr;
                int nc = col + dc;

                if (nr >= 0 && nr < state.rows && nc >= 0 && nc < state.cols) {
                    if (state.board[nr][nc]) count++;
                }
            }
        }
        return count;
    }
}
