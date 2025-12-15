public class GameController implements Runnable {

    private final GameState state;

    public GameController(GameState state) {
        this.state = state;
    }

    @Override
    public void run() {
        state.generation++;
        state.copyNextToCurrent();
        System.out.println("Generación " + state.generation + ":");
        GameState.printBoard(state.board);
    }
}
