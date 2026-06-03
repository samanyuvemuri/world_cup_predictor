from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from services.feature_builder import FeatureBuilder

app = FastAPI(title="World Cup Simulator API")

# load model and initialize the feature builder
model = joblib.load('xgboost_worldcup_model.pkl')
feature_builder = FeatureBuilder()

# load the group structure
with open('data/groups.json', 'r') as f:
    tournament_groups = json.load(f)

# request schema for custom head-to-head
class CustomMatchRequest(BaseModel):
    home_team: str
    away_team: str
    is_neutral: int # one for neutral, zero for true home game

@app.post("/api/head-to-head")
def simulate_single_match(req: CustomMatchRequest):
    # dynamically build the stats based on the team names
    features_dict = feature_builder.build_features(
        req.home_team, 
        req.away_team, 
        req.is_neutral, 
        pd.to_datetime('2026-06-11')
    )
    
    # convert to dataframe in the exact order the model expects
    features_order = [
        'home_elo', 'away_elo', 'elo_diff', 'is_neutral', 
        'home_goals_scored_l5', 'home_goals_allowed_l5', 
        'away_goals_scored_l5', 'away_goals_allowed_l5', 
        'is_competitive', 'home_rest', 'away_rest', 'rest_diff'
    ]
    X_input = pd.DataFrame([features_dict])[features_order]
    
    # get probabilities
    probs = model.predict_proba(X_input)[0]
    
    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "probabilities": {
            "away_win": round(float(probs[0]), 3),
            "draw": round(float(probs[1]), 3),
            "home_win": round(float(probs[2]), 3)
        }
    }

@app.get("/api/simulate/groups")
def simulate_group_stage():
    # explicitly define the column names since the csv doesn't have headers
    col_names = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country', 'neutral']
    schedule = pd.read_csv('data/worldcup.csv', header=None, names=col_names)
    
    # initialize an empty scoreboard for all 48 teams
    scoreboard = {group: {team: {"xPts": 0.0, "matches_played": 0} for team in teams} 
                  for group, teams in tournament_groups.items()}
    
    # loop through all 72 group stage matches
    for _, row in schedule.iterrows():
        # the csv has "neutral" as TRUE/FALSE string. Convert to integer 1 or 0.
        is_neutral = 1 if str(row['neutral']).upper() == 'TRUE' else 0
        
        # build features dynamically
        features = feature_builder.build_features(row['home_team'], row['away_team'], is_neutral, row['date'])
        
        # format for model
        features_order = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral', 'home_goals_scored_l5', 'home_goals_allowed_l5', 'away_goals_scored_l5', 'away_goals_allowed_l5', 'is_competitive', 'home_rest', 'away_rest', 'rest_diff']
        X_input = pd.DataFrame([features])[features_order]
        
        # predict
        probs = model.predict_proba(X_input)[0]
        
        # cast to standard python float to prevent json serialization errors
        home_xpts = float((probs[2] * 3) + (probs[1] * 1))
        away_xpts = float((probs[0] * 3) + (probs[1] * 1))
        
        # find which group these teams are in and assign points
        for group, teams in tournament_groups.items():
            if row['home_team'] in teams:
                scoreboard[group][row['home_team']]["xPts"] += home_xpts
                scoreboard[group][row['home_team']]["matches_played"] += 1
            if row['away_team'] in teams:
                scoreboard[group][row['away_team']]["xPts"] += away_xpts
                scoreboard[group][row['away_team']]["matches_played"] += 1
                
    # round the points to two decimal places to make the JSON cleaner for the frontend
    for group in scoreboard:
        for team in scoreboard[group]:
            scoreboard[group][team]["xPts"] = round(scoreboard[group][team]["xPts"], 2)
            
    return scoreboard