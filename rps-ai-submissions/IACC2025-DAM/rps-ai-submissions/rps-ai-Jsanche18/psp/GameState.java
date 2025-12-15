import java.util.Random;

public class GameState {

    boolean[][] board;
    boolean[][] nextBoard;
    int rows;
    int cols;
    int generation;

    public GameState(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.board = new boolean[rows][cols];
        this.nextBoard = new boolean[rows][cols];
        this.generation = 0;
    }

    public void randomInit() {
        Random random = new Random();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                this.board[r][c] = random.nextInt(100) < 30;
            }
        }
    }

    public void copyNextToCurrent() {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                board[r][c] = nextBoard[r][c];
            }
        }
    }

    public static void printBoard(boolean[][] board) {
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board[0].length; c++) {
                System.out.print(board[r][c] ? "O " : ". ");
            }
            System.out.print("\n");
        }
        System.out.println("------------------------------");
    }
}
