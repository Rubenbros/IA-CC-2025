"""
Script to clean the RPS dataset before training
Removes ALL rows with NONE values (incomplete history)
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('../data/rps_dataset.csv')

print(f"Original dataset: {len(df)} rows")

# Remove ALL rows that contain 'NONE' in ANY column
# This ensures complete history for all features
df_clean = df.copy()

# Check each row for NONE values
mask = ~df_clean.apply(lambda row: row.astype(str).str.contains('NONE').any(), axis=1)
df_clean = df_clean[mask]

print(f"After removing ALL NONE rows: {len(df_clean)} rows")

# Remove duplicate game_numbers (keep first occurrence)
df_clean = df_clean.drop_duplicates(subset=['game_number'], keep='first')

print(f"After removing duplicates: {len(df_clean)} rows")

# Reset index
df_clean = df_clean.reset_index(drop=True)

# Save cleaned dataset
df_clean.to_csv('../data/rps_dataset_clean.csv', index=False)

print(f"\nCleaned dataset saved to: ../data/rps_dataset_clean.csv")
print(f"   Total clean rows: {len(df_clean)}")

# Show data quality
print("\nData Quality Check:")
print(f"   Rows with NONE: {(df_clean == 'NONE').sum().sum()}")
print(f"   Missing values: {df_clean.isnull().sum().sum()}")

print(f"\nMove distribution:")
print(df_clean['p1_current_move'].value_counts())

print(f"\nSample of first 3 rows:")
print(df_clean.head(3)[['game_number', 'p1_last_move', 'p1_last_2_moves', 'p1_last_3_moves', 'p1_current_move']])