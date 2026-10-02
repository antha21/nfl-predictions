import pandas as pd
import numpy as np

# --- CONFIGURATION: YOUR LEAGUE'S SCORING SETTINGS ---
# Adjust these values to match exactly what your league awards!

# 1. Passing Settings
PASS_YD_PT = 1 / 25.0  # 1 point per 25 yards
PASS_TD = 4.0
PASS_INT = -2.0
PASS_COMP = 0.0        # Set to a value if you use Points Per Completion
PASS_INC = 0.0         # Set to negative if you penalize incompletions
PASS_2PT = 2.0
# Passing Bonuses
PASS_TD_40 = 1.0       # Points added for a 40+ yd TD pass
PASS_TD_50 = 2.0       # Points added for a 50+ yd TD pass
PASS_300_GM = 3.0      # Bonus for a 300+ yard passing game
PASS_400_GM = 2.0      # Bonus for a 400+ yard passing game

# 2. Rushing Settings
RUSH_YD_PT = 1 / 10.0  # 1 point per 10 yards
RUSH_TD = 6.0
RUSH_2PT = 2.0
RUSH_ATT = 0.0         # Points Per Carry
# Rushing Bonuses
RUSH_TD_40 = 1.0       # Points added for a 40+ yd TD rush
RUSH_TD_50 = 2.0       # Points added for a 50+ yd TD rush
RUSH_100_GM = 3.0      # Bonus for a 100+ yard rushing game
RUSH_200_GM = 2.0      # Bonus for a 200+ yard rushing game

# 3. Receiving Settings
REC_YD_PT = 1 / 10.0   # 1 point per 10 yards
REC_TD = 6.0
REC_2PT = 2.0
REC_PPR = 1.0          # 1 Point Per Reception (PPR)
# Receiving Bonuses
REC_TD_40 = 1.0        # Points added for a 40+ yd TD catch
REC_TD_50 = 2.0        # Points added for a 50+ yd TD catch
REC_100_GM = 3.0       # Bonus for a 100+ yard receiving game
REC_200_GM = 2.0       # Bonus for a 200+ yard receiving game

# 4. Fumbles
FUM_LOST = -2.0

def build_scoring_engine():
    print("Loading cleaned Play-by-Play dataset...")
    df = pd.read_parquet('nfl_pbp_cleaned.parquet')
    
    # ---------------------------------------------------------
    # STEP 1: SCORE INDIVIDUAL PLAYS
    # ---------------------------------------------------------
    print("Scoring individual passing, rushing, and receiving plays...")
    
    # --- PASSERS ---
    passers = df[df['passer_player_name'].notna()].copy()
    passers['pass_pts'] = (
        (passers['passing_yards'].fillna(0) * PASS_YD_PT) +
        (passers['pass_touchdown'].fillna(0) * PASS_TD) +
        (passers['interception'].fillna(0) * PASS_INT) +
        (passers['complete_pass'].fillna(0) * PASS_COMP) +
        (passers['incomplete_pass'].fillna(0) * PASS_INC)
    )
    # Long TD Bonuses (Checks if the play is a TD AND the yardage is met)
    passers['pass_pts'] += np.where((passers['pass_touchdown'] == 1) & (passers['passing_yards'] >= 40), PASS_TD_40, 0)
    passers['pass_pts'] += np.where((passers['pass_touchdown'] == 1) & (passers['passing_yards'] >= 50), PASS_TD_50, 0)
    # 2PT Conversions
    passers['pass_pts'] += np.where((passers['two_point_attempt'] == 1) & (passers['two_point_conv_result'] == 'success'), PASS_2PT, 0)
    
    # Group the individual plays into weekly totals
    pass_weekly = passers.groupby(['season', 'week', 'passer_player_id', 'passer_player_name']).agg(
        total_pass_pts=('pass_pts', 'sum'),
        total_pass_yds=('passing_yards', 'sum')
    ).reset_index().rename(columns={'passer_player_id': 'player_id', 'passer_player_name': 'player_name'})


    # --- RUSHERS ---
    rushers = df[df['rusher_player_name'].notna()].copy()
    rushers['rush_pts'] = (
        (rushers['rushing_yards'].fillna(0) * RUSH_YD_PT) +
        (rushers['rush_touchdown'].fillna(0) * RUSH_TD) +
        (rushers['rush_attempt'].fillna(0) * RUSH_ATT)
    )
    # Long TD Bonuses
    rushers['rush_pts'] += np.where((rushers['rush_touchdown'] == 1) & (rushers['rushing_yards'] >= 40), RUSH_TD_40, 0)
    rushers['rush_pts'] += np.where((rushers['rush_touchdown'] == 1) & (rushers['rushing_yards'] >= 50), RUSH_TD_50, 0)
    # 2PT Conversions
    rushers['rush_pts'] += np.where((rushers['two_point_attempt'] == 1) & (rushers['two_point_conv_result'] == 'success'), RUSH_2PT, 0)
    
    # Group to weekly totals
    rush_weekly = rushers.groupby(['season', 'week', 'rusher_player_id', 'rusher_player_name']).agg(
        total_rush_pts=('rush_pts', 'sum'),
        total_rush_yds=('rushing_yards', 'sum')
    ).reset_index().rename(columns={'rusher_player_id': 'player_id', 'rusher_player_name': 'player_name'})


    # --- RECEIVERS ---
    receivers = df[df['receiver_player_name'].notna()].copy()
    receivers['rec_pts'] = (
        (receivers['receiving_yards'].fillna(0) * REC_YD_PT) +
        (receivers['pass_touchdown'].fillna(0) * REC_TD) +
        (receivers['complete_pass'].fillna(0) * REC_PPR)
    )
    # Long TD Bonuses
    receivers['rec_pts'] += np.where((receivers['pass_touchdown'] == 1) & (receivers['receiving_yards'] >= 40), REC_TD_40, 0)
    receivers['rec_pts'] += np.where((receivers['pass_touchdown'] == 1) & (receivers['receiving_yards'] >= 50), REC_TD_50, 0)
    # 2PT Conversions
    receivers['rec_pts'] += np.where((receivers['two_point_attempt'] == 1) & (receivers['two_point_conv_result'] == 'success'), REC_2PT, 0)
    
    # Group to weekly totals
    rec_weekly = receivers.groupby(['season', 'week', 'receiver_player_id', 'receiver_player_name']).agg(
        total_rec_pts=('rec_pts', 'sum'),
        total_rec_yds=('receiving_yards', 'sum')
    ).reset_index().rename(columns={'receiver_player_id': 'player_id', 'receiver_player_name': 'player_name'})


    # --- FUMBLES LOST ---
    fumbles = df[(df['fumble_lost'] == 1) & (df['fumbled_1_player_name'].notna())].copy()
    fumbles['fum_pts'] = FUM_LOST
    fum_weekly = fumbles.groupby(['season', 'week', 'fumbled_1_player_id', 'fumbled_1_player_name']).agg(
        total_fum_pts=('fum_pts', 'sum')
    ).reset_index().rename(columns={'fumbled_1_player_id': 'player_id', 'fumbled_1_player_name': 'player_name'})


    # ---------------------------------------------------------
    # STEP 2: COMBINE INTO ONE MASTER SHEET
    # ---------------------------------------------------------
    print("Combining categories into weekly totals...")
    # Merge passing, rushing, receiving, and fumbles together by player and week
    master = pd.merge(pass_weekly, rush_weekly, on=['season', 'week', 'player_id', 'player_name'], how='outer')
    master = pd.merge(master, rec_weekly, on=['season', 'week', 'player_id', 'player_name'], how='outer')
    master = pd.merge(master, fum_weekly, on=['season', 'week', 'player_id', 'player_name'], how='outer')
    
    # Fill any stats a player didn't record (like a WR having 0 passing yards) with 0
    master = master.fillna(0)


    # ---------------------------------------------------------
    # STEP 3: APPLY GAME-LEVEL BONUSES
    # ---------------------------------------------------------
    print("Applying game-level yardage bonuses (100+ yards, 300+ yards)...")
    
    # Passing Bonuses
    master['total_pass_pts'] += np.where(master['total_pass_yds'] >= 300, PASS_300_GM, 0)
    master['total_pass_pts'] += np.where(master['total_pass_yds'] >= 400, PASS_400_GM, 0)
    
    # Rushing Bonuses
    master['total_rush_pts'] += np.where(master['total_rush_yds'] >= 100, RUSH_100_GM, 0)
    master['total_rush_pts'] += np.where(master['total_rush_yds'] >= 200, RUSH_200_GM, 0)
    
    # Receiving Bonuses
    master['total_rec_pts'] += np.where(master['total_rec_yds'] >= 100, REC_100_GM, 0)
    master['total_rec_pts'] += np.where(master['total_rec_yds'] >= 200, REC_200_GM, 0)

    # Final Total Fantasy Points!
    master['weekly_fantasy_points'] = (
        master['total_pass_pts'] + 
        master['total_rush_pts'] + 
        master['total_rec_pts'] + 
        master['total_fum_pts']
    )
    
    # Sort the list so the highest scoring performances are at the top
    master = master.sort_values(by=['season', 'week', 'weekly_fantasy_points'], ascending=[False, False, False])

    print("Success! Saving final scoring file...")
    # We save this as a standard CSV so you can easily open it in Excel if you want to look at it
    master.to_csv('historical_fantasy_scores.csv', index=False)
    print("Saved to 'historical_fantasy_scores.csv'")

if __name__ == '__main__':
    build_scoring_engine()