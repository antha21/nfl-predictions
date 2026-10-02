import nflreadpy as nfl
import pandas as pd

print("Downloading NFL data... This might take a minute or two.")

# Fetch the last three seasons of play-by-play data
# Using range(2021, 2027) tells Python to grab every year from 2021 up to 2026.
pbp_raw = nfl.load_pbp(range(2021, 2027)).to_pandas()
print("Data downloaded! Filtering out noise (penalties, kneels, spikes)...")


# Keep only regular season and playoff games
pbp_clean = pbp_raw[pbp_raw['season_type'].isin(['REG', 'POST'])]

# Isolate actual offensive snaps
valid_plays = pbp_clean[
    (pbp_clean['play_type'].isin(['pass', 'run'])) & 
    (pbp_clean['qb_kneel'] == 0) & 
    (pbp_clean['qb_spike'] == 0) &
    (pbp_clean['penalty'] == 0)
]

print(f"Filtering complete. Saving {len(valid_plays)} valid plays to a Parquet file...")

# Save the cleaned dataframe
valid_plays.to_parquet('nfl_pbp_cleaned.parquet')

print("Success! Check your folder for 'nfl_pbp_cleaned.parquet'.")