import pandas as pd

df = pd.read_csv('rps_dataset.csv')

# Check row 2 and 3 (actually rows 3 and 4 in CSV due to header)
print("\n=== Testing for Data Leakage ===\n")

# Get a few examples
for i in range(2, 7):
    current_row = df.iloc[i]
    next_row = df.iloc[i + 1] if i + 1 < len(df) else None

    print(f"Game {current_row['game_number']}:")
    print(f"  Current move: {current_row['p1_current_move']}")

    if next_row is not None:
        print(f"  Next game's last_move: {next_row['p1_last_move']}")
        print(f"  Match: {current_row['p1_current_move'] == next_row['p1_last_move']}")
    print()

print("If all matches are True, there's NO data leakage!")