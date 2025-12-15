"""
Rock Paper Scissors - Data Collection Script
Two humans play against each other to generate training data
"""

import csv
import os
from collections import deque


class GameDataCollector:
    """Collects game data from two human players"""

    def __init__(self, filename='../data/rps_dataset.csv', player1_name='Player1', player2_name='Player2'):
        self.filename = filename
        self.player1_history = deque(maxlen=10)
        self.player2_history = deque(maxlen=10)
        self.outcomes_p1 = deque(maxlen=10)  # win/lose/tie from player 1's perspective
        self.game_count = 0
        self.data_rows_saved = 0  # Track actual rows written to CSV

        # Track total wins/losses/ties across all games
        self.total_wins = 0
        self.total_losses = 0
        self.total_ties = 0

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(filename):
            self.create_csv()
        else:
            # If file exists, restore state from CSV
            self._restore_state_from_csv()

    def _restore_state_from_csv(self):
        """Restore game state from existing CSV file"""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

                if len(lines) <= 1:  # Only header or empty
                    return

                # Get the last up to 10 rows to restore history
                data_lines = [line.strip() for line in lines[1:] if line.strip()]
                last_rows = data_lines[-10:] if len(data_lines) > 10 else data_lines

                # Parse each row and restore history
                for i, line in enumerate(last_rows):
                    parts = line.split(',')
                    if len(parts) >= 17:
                        p1_move = parts[16]  # p1_current_move (last column)

                        # To get p2's move, we need to look at the NEXT row's p2_last_move
                        # OR use p1_last_outcome to infer what p2 played
                        # But the easiest is: if we have the next row, use its p2_last_move
                        p2_move = None

                        if i + 1 < len(last_rows):
                            # Get p2's move from the next row's p2_last_move
                            next_parts = last_rows[i + 1].split(',')
                            if len(next_parts) >= 17:
                                p2_move = next_parts[10]  # p2_last_move from next row
                        else:
                            # This is the last row, so we don't have p2's move yet
                            # We'll need to infer it from the outcome or skip it
                            p1_last_outcome = parts[9]  # p1_last_outcome
                            if p1_last_outcome != 'NONE':
                                # Try to infer p2's move from the outcome
                                p2_move = self._infer_p2_move(p1_move, p1_last_outcome)

                        # Only add if not NONE and we have both moves
                        if p1_move != 'NONE' and p2_move and p2_move != 'NONE':
                            self.player1_history.append(p1_move)
                            self.player2_history.append(p2_move)

                            # Determine outcome for this move
                            outcome = self.determine_outcome(p1_move, p2_move)
                            self.outcomes_p1.append(outcome)

                # Set game_count to continue from last game
                if data_lines:
                    last_line = data_lines[-1]
                    last_game_num = int(last_line.split(',')[0])
                    self.game_count = last_game_num + 1

                print(f"\nRestored state from CSV:")
                print(f"  - Starting from game {self.game_count}")
                print(f"  - Loaded last {len(self.player1_history)} moves into history")

        except Exception as e:
            print(f"Could not restore state from CSV: {e}")
            print("Starting fresh...")
            self.game_count = 0

    def _infer_p2_move(self, p1_move, outcome):
        """Infer what Player 2 played based on Player 1's move and outcome"""
        if outcome == 'tie':
            return p1_move
        elif outcome == 'win':
            # P1 won, so P2 played what P1 beats
            beats = {'R': 'S', 'P': 'R', 'S': 'P'}
            return beats[p1_move]
        elif outcome == 'lose':
            # P1 lost, so P2 played what beats P1
            beaten_by = {'R': 'P', 'P': 'S', 'S': 'R'}
            return beaten_by[p1_move]
        return None

    def create_csv(self):
        """Create CSV file with headers"""
        headers = [
            'game_number',
            # Player 1 features
            'p1_last_move',
            'p1_last_2_moves',
            'p1_last_3_moves',
            'p1_rock_freq',
            'p1_paper_freq',
            'p1_scissors_freq',
            'p1_win_streak',
            'p1_loss_streak',
            'p1_last_outcome',
            # Player 2 features (opponent)
            'p2_last_move',
            'p2_last_2_moves',
            'p2_last_3_moves',
            'p2_rock_freq',
            'p2_paper_freq',
            'p2_scissors_freq',
            # Target variable
            'p1_current_move'
        ]

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        print(f"Created dataset file: {self.filename}")

    def get_move_frequencies(self, history):
        """Calculate move frequencies"""
        if len(history) == 0:
            return 0.33, 0.33, 0.33

        total = len(history)
        rock_freq = history.count('R') / total
        paper_freq = history.count('P') / total
        scissors_freq = history.count('S') / total

        return rock_freq, paper_freq, scissors_freq

    def get_streak(self, outcomes, outcome_type):
        """Calculate current win or loss streak"""
        streak = 0
        for outcome in reversed(outcomes):
            if outcome == outcome_type:
                streak += 1
            else:
                break
        return streak

    def get_last_n_moves(self, history, n):
        """Get last N moves as a string"""
        if len(history) < n:
            return 'NONE'
        return ''.join(list(history)[-n:])

    def determine_outcome(self, p1_move, p2_move):
        """Determine outcome from player 1's perspective"""
        if p1_move == p2_move:
            return 'tie'
        wins = {'R': 'S', 'P': 'R', 'S': 'P'}
        return 'win' if wins[p1_move] == p2_move else 'lose'

    def record_round(self, p1_move, p2_move):
        """Record a single round of play"""
        # Save data even for the first game (game_count starts at 0)
        # For the first game, all history features will be 'NONE' or default values

        # Extract features BEFORE adding current move
        p1_last = self.player1_history[-1] if len(self.player1_history) >= 1 else 'NONE'
        p1_last_2 = self.get_last_n_moves(self.player1_history, 2)
        p1_last_3 = self.get_last_n_moves(self.player1_history, 3)
        p1_rock_freq, p1_paper_freq, p1_scissors_freq = self.get_move_frequencies(self.player1_history)
        p1_win_streak = self.get_streak(self.outcomes_p1, 'win')
        p1_loss_streak = self.get_streak(self.outcomes_p1, 'lose')
        p1_last_outcome = self.outcomes_p1[-1] if len(self.outcomes_p1) >= 1 else 'NONE'

        p2_last = self.player2_history[-1] if len(self.player2_history) >= 1 else 'NONE'
        p2_last_2 = self.get_last_n_moves(self.player2_history, 2)
        p2_last_3 = self.get_last_n_moves(self.player2_history, 3)
        p2_rock_freq, p2_paper_freq, p2_scissors_freq = self.get_move_frequencies(self.player2_history)

        # Save to CSV
        row = [
            self.game_count,
            p1_last,
            p1_last_2,
            p1_last_3,
            p1_rock_freq,
            p1_paper_freq,
            p1_scissors_freq,
            p1_win_streak,
            p1_loss_streak,
            p1_last_outcome,
            p2_last,
            p2_last_2,
            p2_last_3,
            p2_rock_freq,
            p2_paper_freq,
            p2_scissors_freq,
            p1_move  # This is what we're trying to predict
        ]

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.data_rows_saved += 1

        # Update histories AFTER saving
        outcome = self.determine_outcome(p1_move, p2_move)
        self.player1_history.append(p1_move)
        self.player2_history.append(p2_move)
        self.outcomes_p1.append(outcome)
        self.game_count += 1

        # Update total counts
        if outcome == 'win':
            self.total_wins += 1
        elif outcome == 'lose':
            self.total_losses += 1
        else:
            self.total_ties += 1

        return outcome

    def get_stats(self):
        """Get current game statistics"""
        if self.game_count == 0:
            return "No games played yet"

        return f"Games: {self.game_count} | P1 Wins: {self.total_wins} | P2 Wins: {self.total_losses} | Ties: {self.total_ties} | Rows saved: {self.data_rows_saved}"


def play_data_collection():
    """Main game loop for data collection"""
    print("=" * 60)
    print("Rock Paper Scissors - DATA COLLECTION MODE")
    print("=" * 60)
    print("\nThis mode allows two humans to play against each other.")
    print("All games will be saved to create a training dataset.")
    print("\nCommands:")
    print("  R, r, rock - Play Rock")
    print("  P, p, paper - Play Paper")
    print("  S, s, scissors - Play Scissors")
    print("  stats - Show statistics")
    print("  quit - Exit and save data")
    print("\n" + "=" * 60)

    collector = GameDataCollector()

    while True:
        print(f"\n{collector.get_stats()}")
        print("\n--- New Round ---")

        # Player 1's move
        p1_input = input("Player 1, your move: ").strip().lower()

        if p1_input in ['quit', 'q', 'exit']:
            print(f"\nData collection complete!")
            print(f"Total games played: {collector.game_count}")
            print(f"Total rows saved to CSV: {collector.data_rows_saved}")
            print(f"Dataset saved to: {collector.filename}")
            break

        if p1_input == 'stats':
            continue

        # Parse Player 1's move
        move_map = {
            'r': 'R', 'rock': 'R',
            'p': 'P', 'paper': 'P',
            's': 'S', 'scissors': 'S'
        }

        if p1_input not in move_map:
            print("Invalid input! Use R/P/S or rock/paper/scissors")
            continue

        p1_move = move_map[p1_input]

        # Player 2's move
        p2_input = input("Player 2, your move: ").strip().lower()

        if p2_input in ['quit', 'q', 'exit']:
            print(f"\nData collection complete!")
            print(f"Total games played: {collector.game_count}")
            print(f"Total rows saved to CSV: {collector.data_rows_saved}")
            print(f"Dataset saved to: {collector.filename}")
            break

        if p2_input not in move_map:
            print("Invalid input! Use R/P/S or rock/paper/scissors")
            continue

        p2_move = move_map[p2_input]

        # Record the round
        outcome = collector.record_round(p1_move, p2_move)

        # Display result
        move_names = {'R': 'Rock', 'P': 'Paper', 'S': 'Scissors'}
        print(f"\nPlayer 1 played: {move_names[p1_move]}")
        print(f"Player 2 played: {move_names[p2_move]}")

        if outcome == 'win':
            print("Result: Player 1 WINS!")
        elif outcome == 'lose':
            print("Result: Player 2 WINS!")
        else:
            print("Result: TIE!")


if __name__ == "__main__":
    play_data_collection()