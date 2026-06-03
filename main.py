import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# loads both of my csv files
df_matches = pd.read_csv('results.csv')
df_elo = pd.read_csv('eloratings.csv')

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
    filtered_elo[['date', 'team', 'rating', 'elo_momentum']].rename(columns={'team': 'home_team'}),
    on='date', 
    by='home_team', 
    direction='backward'
).rename(columns={'rating': 'home_elo', 'elo_momentum': 'home_elo_momentum'})

df_merge = pd.merge_asof(
    df_merge, 
    filtered_elo[['date', 'team', 'rating', 'elo_momentum']].rename(columns={'team': 'away_team'}),
    on='date', 
    by='away_team', 
    direction='backward'
).rename(columns={'rating': 'away_elo', 'elo_momentum': 'away_elo_momentum'})

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

# calculate rolling averages for the last 5 overall games (excluding the current match)
team_history['goals_scored_l5'] = team_history.groupby('team')['goals_scored'].transform(lambda x: x.rolling(5, closed='left').mean())
team_history['goals_allowed_l5'] = team_history.groupby('team')['goals_allowed'].transform(lambda x: x.rolling(5, closed='left').mean())

# drop the redundant columns before merging back to keep things clean
team_history = team_history[['date', 'team', 'goals_scored_l5', 'goals_allowed_l5']]

# merge these overall stats back into your main dataframe
# merge for the home team
df_merge = pd.merge(
    df_merge, 
    team_history.rename(columns={
        'team': 'home_team', 
        'goals_scored_l5': 'home_goals_scored_l5', 
        'goals_allowed_l5': 'home_goals_allowed_l5'
    }),
    on=['date', 'home_team'],
    how='left'
)

# merge for the away team
df_merge = pd.merge(
    df_merge, 
    team_history.rename(columns={
        'team': 'away_team', 
        'goals_scored_l5': 'away_goals_scored_l5', 
        'goals_allowed_l5': 'away_goals_allowed_l5'
    }),
    on=['date', 'away_team'],
    how='left'
)

# filter for the "modern" era of football, made the cutoff ~20 years ago
# filters to ~20k rows
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
    'is_competitive', 'home_elo_momentum', 'away_elo_momentum'
]
df_modern = df_modern.dropna(subset=features)
X = df_modern[features]
y = df_modern['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) #my goat bijan

# fitting random forest classifier
model = RandomForestClassifier(n_estimators=150, random_state=7)
model.fit(X_train, y_train)

# model evaluation: random forest classifier
prediction = model.predict(X_test)
print(f"RandomForest Baseline Accuracy: {accuracy_score(y_test, prediction):.2%}\n")
print(classification_report(y_test, prediction, target_names=['Away Win', 'Draw', 'Home Win']))

# results:
# model baseline accuracy: 57.21%

#              precision    recall  f1-score   support
#
#    away win       0.54      0.55      0.55      1009
#        draw       0.32      0.14      0.20       847
#    home win       0.63      0.79      0.70      1729

#    accuracy                           0.57      3585
#   macro avg       0.50      0.50      0.48      3585
#     wtd avg       0.53      0.57      0.54      3585

#######################################################################################################################

# fitting xgboost classifier

# define custom weight factors
weight_dict = {
    0: 1.0,
    1: 1.4,
    2: 1.0,
}

# map these weights to training set targets
sample_weights = y_train.map(weight_dict)

# pass the weights into the fitting process
model = XGBClassifier(
    n_estimators=180, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=7,
    objective='multi:softmax',
    num_class=3
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# model evaluation: xgboost classifier
prediction = model.predict(X_test)
print(f"XGBoost Baseline Accuracy: {accuracy_score(y_test, prediction):.2%}\n")
print(classification_report(y_test, prediction, target_names=['Away Win', 'Draw', 'Home Win']))

# results:
# model baseline accuracy: 58.72%

#              precision    recall  f1-score   support
#
#    away win       0.54      0.59      0.56      1009
#        draw       0.36      0.04      0.07       847
#    home win       0.62      0.85      0.72      1729

#    accuracy                           0.59      3585
#   macro avg       0.51      0.49      0.45      3585
#     wtd avg       0.54      0.59      0.52      3585