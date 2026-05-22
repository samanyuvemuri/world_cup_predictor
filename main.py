import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# filter for the "modern" era of football, made the cutoff ~20 years ago
# filters to ~20k rows
df_modern = df_merge[df_merge['date'] > '2006-01-01']

# quantifying a team's recent offensive form
df_modern = df_modern.sort_values('date')
df_modern['home_goals_l5'] = df_modern.groupby('home_team')['home_score'].transform(lambda x: x.rolling(5, closed='left').mean())
df_modern['away_goals_l5'] = df_modern.groupby('away_team')['away_score'].transform(lambda x: x.rolling(5, closed='left').mean())

# 2 = fifa world cup match, 1 = other, 0 = friendly exhibition
conditions = [
    df_modern['tournament'].str.contains('FIFA World Cup', case=False, na=False),
    df_modern['tournament'].str.contains('Friendly', case=False, na=False)
]
choices = [2, 0]
df_modern['is_competitive'] = np.select(conditions, choices, default=1)

# train / test split
features = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral', 'home_goals_l5', 'away_goals_l5', 'is_competitive']
X = df_modern[features]
y = df_modern['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) #my goat bijan

# fitting random forest classifier
model = RandomForestClassifier(n_estimators=150, random_state=7)
model.fit(X_train, y_train)

# model evaluation: random forest classifier
prediction = model.predict(X_test)
print(f"Model Baseline Accuracy: {accuracy_score(y_test, prediction):.2%}\n")
print(classification_report(y_test, prediction, target_names=['Away Win', 'Draw', 'Home Win']))
print(df_modern)
# results:
# model baseline accuracy: 56.95%

#              precision    recall  f1-score   support
#
#    away win       0.55      0.55      0.55      1032
#        draw       0.28      0.15      0.19       833
#    home win       0.64      0.78      0.71      1738

#    accuracy                           0.57      3603
#   macro avg       0.49      0.49      0.48      3603
#     wtd avg       0.53      0.57      0.54      3603

#######################################################################################################################

# implement xgboost next to see if accuracy increases