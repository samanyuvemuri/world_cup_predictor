import pandas as pd
import numpy as np
from datetime import datetime

class FeatureBuilder:
    def __init__(self, elo_path='data/eloratings.csv', results_path='data/results.csv'):
        # load and clean the historical data exactly like your training script
        self.df_elo = pd.read_csv(elo_path)
        self.df_elo['date'] = pd.to_datetime(self.df_elo['date'])
        
        # clean team names exactly as before
        name_map = {'China': 'China PR', 'Czechia': 'Czech Republic', 'Democratic Republic of Congo': 'DR Congo', 'Ireland': 'Republic of Ireland', 'Sao Tome and Principe': 'São Tomé and Príncipe', 'East Timor': 'Timor-Leste', 'US Virgin Islands': 'United States Virgin Islands'}
        self.df_elo['team'] = self.df_elo['team'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().replace(name_map)
        self.df_elo = self.df_elo.sort_values(['team', 'date'])

        self.df_results = pd.read_csv(results_path)
        self.df_results['date'] = pd.to_datetime(self.df_results['date'])
        
        # set the anchor date for the start of the world cup
        self.anchor_date = pd.to_datetime('2026-06-11')

    def get_latest_elo(self, team_name):
        # look backward from the start of the world cup to find their last known elo
        team_elos = self.df_elo[(self.df_elo['team'] == team_name) & (self.df_elo['date'] < self.anchor_date)]
        if team_elos.empty:
            return 1500.0 # default if missing
        return team_elos.iloc[-1]['rating']

    def get_form_l5(self, team_name):
        # get the last five matches played by the team before the world cup
        home_matches = self.df_results[(self.df_results['home_team'] == team_name) & (self.df_results['date'] < self.anchor_date)]
        away_matches = self.df_results[(self.df_results['away_team'] == team_name) & (self.df_results['date'] < self.anchor_date)]
        
        home_history = home_matches[['date', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'scored', 'away_score': 'allowed'})
        away_history = away_matches[['date', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'scored', 'home_score': 'allowed'})
        
        history = pd.concat([home_history, away_history]).sort_values('date').tail(5)
        
        if history.empty:
            return 1.0, 1.0 # default if no data
            
        return history['scored'].mean(), history['allowed'].mean()

    def build_features(self, home_team, away_team, is_neutral, match_date):
        # get elo ratings
        home_elo = self.get_latest_elo(home_team)
        away_elo = self.get_latest_elo(away_team)
        
        # get recent form
        home_scored, home_allowed = self.get_form_l5(home_team)
        away_scored, away_allowed = self.get_form_l5(away_team)
        
        # calculate rest difference
        # for simplicity in the app, we assume everyone is rested for their first match.
        # in a highly advanced version, you would calculate exact days between group stage matches.
        # we will use a standard 5 days rest for mid-tournament group games.
        home_rest = 5.0
        away_rest = 5.0
        
        # construct the exact feature dictionary your model expects
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "is_neutral": int(is_neutral),
            "home_goals_scored_l5": home_scored,
            "home_goals_allowed_l5": home_allowed,
            "away_goals_scored_l5": away_scored,
            "away_goals_allowed_l5": away_allowed,
            "is_competitive": 2, # always two for world cup
            "home_rest": home_rest,
            "away_rest": away_rest,
            "rest_diff": home_rest - away_rest
        }