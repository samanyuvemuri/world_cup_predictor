from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# initialize the app and load the model
app = FastAPI(title="World Cup Predictor API")
model = joblib.load('xgboost_worldcup_model.pkl')

# define the expected incoming data structure
class MatchRequest(BaseModel):
    home_elo: float
    away_elo: float
    elo_diff: float
    is_neutral: int
    home_goals_scored_l5: float
    home_goals_allowed_l5: float
    away_goals_scored_l5: float
    away_goals_allowed_l5: float
    is_competitive: int
    home_rest: float
    away_rest: float
    rest_diff: float

# create the prediction endpoint
@app.post("/predict")
def predict_match(match: MatchRequest):
    # convert incoming json payload to a dataframe
    input_data = pd.DataFrame([match.model_dump() if hasattr(match, 'model_dump') else match.dict()])  

    # ensure the columns match exactly what x_train had
    features = [
        'home_elo', 'away_elo', 'elo_diff', 'is_neutral', 
        'home_goals_scored_l5', 'home_goals_allowed_l5', 
        'away_goals_scored_l5', 'away_goals_allowed_l5', 
        'is_competitive', 'home_rest', 'away_rest', 'rest_diff'
    ]
    X_input = input_data[features]
    
    # get probabilities
    probs = model.predict_proba(X_input)[0]
    
    # calculate expected points
    home_xpts = (probs[2] * 3) + (probs[1] * 1)
    away_xpts = (probs[0] * 3) + (probs[1] * 1)
    
    return {
        "probabilities": {
            "away_win": round(float(probs[0]), 3),
            "draw": round(float(probs[1]), 3),
            "home_win": round(float(probs[2]), 3)
        },
        "expected_points": {
            "home": round(float(home_xpts), 2),
            "away": round(float(away_xpts), 2)
        }
    }