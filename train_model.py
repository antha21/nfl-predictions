import pandas as pd
import numpy as np
import xgboost as xgb
import nflreadpy as nfl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_fantasy_model():
    print("1. Loading custom fantasy scores...")
    df = pd.read_csv('historical_fantasy_scores.csv')

    print("2. Fetching schedule and opponent data from nflverse...")
    # Load basic player stats just to easily grab what team they were on and who they played
    stats = nfl.load_player_stats(range(2021, 2027)).to_pandas()
    team_map = stats[['season', 'week', 'player_id', 'team', 'opponent_team']].drop_duplicates()
    
    # Merge the opponent data into our custom scoring sheet
    df = df.merge(team_map, on=['season', 'week', 'player_id'], how='inner')

    print("3. Engineering Machine Learning Features...")
    # Sort the data chronologically so our rolling averages work correctly
    df = df.sort_values(by=['player_id', 'season', 'week'])

    # FEATURE 1: The Player's Baseline (Rolling 3-game average)
    # This teaches the model how "good" the player is right now, ignoring the past seasons.
    df['player_rolling_pts'] = df.groupby('player_id')['weekly_fantasy_points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # FEATURE 2: The Defensive Vulnerability
    # Calculate how many points the opponent's defense has given up on average this season
    defensive_stats = df.groupby(['season', 'week', 'opponent_team'])['weekly_fantasy_points'].sum().reset_index()
    defensive_stats = defensive_stats.sort_values(by=['opponent_team', 'season', 'week'])
    
    # Calculate the defense's rolling average points allowed heading into the week
    defensive_stats['def_rolling_pts_allowed'] = defensive_stats.groupby(['opponent_team', 'season'])['weekly_fantasy_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Merge the defensive vulnerability feature back into our main dataset
    defensive_stats = defensive_stats[['season', 'week', 'opponent_team', 'def_rolling_pts_allowed']]
    df = df.merge(defensive_stats, on=['season', 'week', 'opponent_team'], how='left')

    # Drop any rows with missing data (like Week 1 games where there is no rolling average yet)
    df = df.dropna(subset=['player_rolling_pts', 'def_rolling_pts_allowed'])

    print("4. Splitting data into Training and Testing sets...")
    # X = The Features (What the model looks at to make a guess)
    X = df[['player_rolling_pts', 'def_rolling_pts_allowed']]
    
    # y = The Target (The actual fantasy points they scored that week)
    y = df['weekly_fantasy_points']

    # We hold back 20% of the data to "test" the model after it learns.
    # We do not shuffle the data, because we want to train on the past to predict the future.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("5. Training the XGBoost Model...")
    # We use an XGBoost Regressor because we are predicting a continuous number (points), not a category
    model = xgb.XGBRegressor(
        n_estimators=100,        # Build 100 sequential decision trees
        learning_rate=0.1,       # Take small steps to prevent overfitting
        max_depth=4,             # Limit the tree to 4 questions deep
        objective='reg:squarederror' # Minimize the Mean Squared Error (The Loss Function)
    )

    model.fit(X_train, y_train)

    print("6. Evaluating the Model...")
    # Force the model to predict the 20% of data it was not allowed to look at
    predictions = model.predict(X_test)
    
    # Calculate how far off the model is on average
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nSUCCESS! The model is trained.")
    print(f"On average, the model's predictions are off by: {mae:.2f} fantasy points per game.")
    
    # Let's show a quick example of what it learned!
    # A dummy scenario: A player averaging 20 points, facing a defense giving up 80 total points
    sample_feature = pd.DataFrame({'player_rolling_pts': [20.0], 'def_rolling_pts_allowed': [80.0]})
    sample_pred = model.predict(sample_feature)[0]
    print(f"\nSimulation: A player averaging 20 pts/game faces a tough defense allowing only 80 total pts/game.")
    print(f"XGBoost mathematically projects them to score: {sample_pred:.2f} points.")

if __name__ == '__main__':
    train_fantasy_model()