import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# loads both of my csv files
df_matches = pd.read_csv('data/results.csv')
df_elo = pd.read_csv('data/eloratings.csv')

# converts dates to actual datetime
df_matches['date'] = pd.to_datetime(df_matches['date'])
df_elo['date'] = pd.to_datetime(df_elo['date'])

# filters out columns that i don't want to use
matches_selected_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral']
elo_selected_cols = ['date', 'team', 'rating']
filtered_matches = df_matches[matches_selected_cols]
filtered_elo = df_elo[elo_selected_cols]

filtered_matches = filtered_matches.sort_values('date')
filtered_elo = filtered_elo.sort_values('date')

# eloratings data had a weird non-breaking space instead of a normal one
filtered_elo['team'] = filtered_elo['team'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

# fixing up team names that don't match up between both csv files
name_map = {
    'China': 'China PR',
    'Czechia': 'Czech Republic',
    'Democratic Republic of Congo': 'DR Congo',
    'Ireland': 'Republic of Ireland',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'East Timor': 'Timor-Leste',
    'US Virgin Islands': 'United States Virgin Islands'
}
filtered_elo['team'] = filtered_elo['team'].replace(name_map)

# sort by team and then by date to ensure proper chronological shifting
filtered_elo = filtered_elo.sort_values(['team', 'date'])

# grab the rating from exactly five games ago for each team
filtered_elo['rating_5_ago'] = filtered_elo.groupby('team')['rating'].shift(5)

# calculate momentum (positive means they are hot, negative means they are slumping)
filtered_elo['elo_momentum'] = filtered_elo['rating'] - filtered_elo['rating_5_ago']

# sort back by date so merge_asof works correctly
filtered_elo = filtered_elo.sort_values('date')

# merging the last recorded elo rating at the time of each match to its corresponding home and away teams
df_merge = pd.merge_asof(
    filtered_matches, 
    filtered_elo[['date', 'team', 'rating']].rename(columns={'team': 'home_team'}),
    on='date', 
    by='home_team', 
    direction='backward'
).rename(columns={'rating': 'home_elo'})

df_merge = pd.merge_asof(
    df_merge, 
    filtered_elo[['date', 'team', 'rating']].rename(columns={'team': 'away_team'}),
    on='date', 
    by='away_team', 
    direction='backward'
).rename(columns={'rating': 'away_elo'})

# removing data without any elo ratings, before 1901
df_merge = df_merge.dropna(subset=['home_elo', 'away_elo'])

# 2 = home team win, 1 = draw, 0 = away team win
def get_outcome(row):
    if row['home_score'] > row['away_score']: return 2
    elif row['home_score'] == row['away_score']: return 1
    else: return 0
df_merge['outcome'] = df_merge.apply(get_outcome, axis=1)

# core math calculations for features
df_merge['elo_diff'] = df_merge['home_elo'] - df_merge['away_elo']
df_merge['is_neutral'] = df_merge['neutral'].astype(int)

# create a unified timeline of all matches for every team
home_history = df_merge[['date', 'home_team', 'home_score', 'away_score']].copy()
home_history.columns = ['date', 'team', 'goals_scored', 'goals_allowed']

away_history = df_merge[['date', 'away_team', 'away_score', 'home_score']].copy()
away_history.columns = ['date', 'team', 'goals_scored', 'goals_allowed']

# combine, sort by team and then by date to ensure chronological order
team_history = pd.concat([home_history, away_history]).sort_values(['team', 'date'])

# calculate rolling averages for form
team_history['goals_scored_l5'] = team_history.groupby('team')['goals_scored'].transform(lambda x: x.rolling(5, closed='left').mean())
team_history['goals_allowed_l5'] = team_history.groupby('team')['goals_allowed'].transform(lambda x: x.rolling(5, closed='left').mean())

# calculate schedule congestion
team_history['prev_match_date'] = team_history.groupby('team')['date'].shift(1)
team_history['rest_days'] = (team_history['date'] - team_history['prev_match_date']).dt.days
team_history['rest_days'] = team_history['rest_days'].fillna(7).clip(upper=14)

# drop redundant columns before merging
team_history = team_history[['date', 'team', 'goals_scored_l5', 'goals_allowed_l5', 'rest_days']]

# merge these overall stats back into your main dataframe
df_merge = pd.merge(
    df_merge, 
    team_history.rename(columns={
        'team': 'home_team', 
        'goals_scored_l5': 'home_goals_scored_l5', 
        'goals_allowed_l5': 'home_goals_allowed_l5',
        'rest_days': 'home_rest'
    }),
    on=['date', 'home_team'],
    how='left'
)

df_merge = pd.merge(
    df_merge, 
    team_history.rename(columns={
        'team': 'away_team', 
        'goals_scored_l5': 'away_goals_scored_l5', 
        'goals_allowed_l5': 'away_goals_allowed_l5',
        'rest_days': 'away_rest'
    }),
    on=['date', 'away_team'],
    how='left'
)

# calculate the final rest difference feature
df_merge['rest_diff'] = df_merge['home_rest'] - df_merge['away_rest']

# filter for the "modern" era of football, made the cutoff ~20 years ago
df_modern = df_merge[df_merge['date'] > '2006-01-01']

# 2 = fifa world cup match, 1 = other, 0 = friendly exhibition
conditions = [
    df_modern['tournament'].str.contains('FIFA World Cup', case=False, na=False),
    df_modern['tournament'].str.contains('Friendly', case=False, na=False)
]
choices = [2, 0]
df_modern['is_competitive'] = np.select(conditions, choices, default=1)

# train / test split
features = [
    'home_elo', 'away_elo', 'elo_diff', 'is_neutral', 
    'home_goals_scored_l5', 'home_goals_allowed_l5', 
    'away_goals_scored_l5', 'away_goals_allowed_l5', 
    'is_competitive', 'home_rest', 'away_rest', 'rest_diff'
]
df_modern = df_modern.dropna(subset=features)
X = df_modern[features]
y = df_modern['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) #my goat bijan

# fitting xgboost classifier
weight_dict = {
    0: 1.0,
    1: 1.5,
    2: 1.0,
}

sample_weights = y_train.map(weight_dict)

model = XGBClassifier(
    n_estimators=180, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=7,
    objective='multi:softmax',
    num_class=3
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# model evaluation
prediction = model.predict(X_test)
print(f"XGBoost Baseline Accuracy: {accuracy_score(y_test, prediction):.2%}\n")
print(classification_report(y_test, prediction, target_names=['Away Win', 'Draw', 'Home Win']))

# results:
# xgboost baseline accuracy: 57.13%

#              precision    recall  f1-score   support
#
#    away win       0.60      0.47      0.53      1009
#        draw       0.32      0.32      0.32       847
#    home win       0.67      0.75      0.71      1729

#    accuracy                           0.57      3585
#   macro avg       0.53      0.52      0.52      3585
#     wtd avg       0.57      0.57      0.57      3585

# save the model for the api
joblib.dump(model, 'xgboost_worldcup_model.pkl')
print("Model saved successfully as xgboost_worldcup_model.pkl")