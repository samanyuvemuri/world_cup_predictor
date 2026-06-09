import pandas as pd
import numpy as np

class MonteCarloEngine:
    def __init__(self, tournament_groups, schedule, model, feature_builder):
        self.groups = tournament_groups
        self.schedule = schedule
        self.model = model
        self.feature_builder = feature_builder
        self.match_cache = {}

    def _build_probability_cache(self):
        for index, row in self.schedule.iterrows():
            is_neutral = 1 if str(row['neutral']).upper() == 'TRUE' else 0
            
            features = self.feature_builder.build_features(
                row['home_team'], row['away_team'], is_neutral, row['date']
            )
            features_order = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral', 'home_goals_scored_l5', 
                              'home_goals_allowed_l5', 'away_goals_scored_l5', 'away_goals_allowed_l5', 
                              'is_competitive', 'home_rest', 'away_rest', 'rest_diff']
            
            X_input = pd.DataFrame([features])[features_order]
            probs = self.model.predict_proba(X_input)[0]
            
            # convert to float64 and force normalization to strictly sum to 1.0
            prob_array = np.array([probs[0], probs[1], probs[2]], dtype=np.float64)
            prob_array /= prob_array.sum() 
            
            match_id = f"{row['home_team']}_vs_{row['away_team']}"
            self.match_cache[match_id] = prob_array

    def run_simulation(self, iterations=10000):
        # pre-calculate all match probabilities
        if not self.match_cache:
            self._build_probability_cache()

        # setup the tracker for how many times each team reaches the round of 32
        # start everyone at 0%
        teams = [team for group in self.groups.values() for team in group]
        results_tracker = {team: {"R32_Appearances": 0} for team in teams}

        # 3. The Monte Carlo Loop
        for _ in range(iterations):
            # temporary scoreboard for this specific simulation universe
            sim_scoreboard = {team: 0 for team in teams}
            
            for index, row in self.schedule.iterrows():
                match_id = f"{row['home_team']}_vs_{row['away_team']}"
                probs = self.match_cache[match_id]
                
                # roll the dice, outcomes: 0 (loss), 1 (draw), 3 (win)
                # p=probs passes the xgboost probabilities to weight the dice
                home_points = np.random.choice([0, 1, 3], p=probs)
                
                if home_points == 3:
                    sim_scoreboard[row['home_team']] += 3
                elif home_points == 1:
                    sim_scoreboard[row['home_team']] += 1
                    sim_scoreboard[row['away_team']] += 1
                else:
                    sim_scoreboard[row['away_team']] += 3

            # determine who advances in the current iteration
            advancing_teams = self._calculate_advancement(sim_scoreboard)
            
            for team in advancing_teams:
                results_tracker[team]["R32_Appearances"] += 1

        # convert raw counts to percentages
        final_probabilities = {}
        for team, stats in results_tracker.items():
            prob = (stats["R32_Appearances"] / iterations) * 100
            final_probabilities[team] = {"Reach_R32_Percent": round(prob, 2)}

        return final_probabilities

    def _calculate_advancement(self, sim_scoreboard):
        """Helper function to find the top 2 from each group + top 8 3rd place teams."""
        advancing = []
        third_place_pool = []

        for group_name, group_teams in self.groups.items():
            # sort teams in the group by their points in this simulation
            group_standings = sorted([(team, sim_scoreboard[team]) for team in group_teams], 
                                     key=lambda x: x[1], reverse=True)
            
            # Top two advance automatically
            advancing.append(group_standings[0][0])
            advancing.append(group_standings[1][0])
            
            # third place goes to the waiting pool
            third_place_pool.append(group_standings[2])

        # sort the twelve third-place teams by points and take the top eight
        third_place_pool.sort(key=lambda x: x[1], reverse=True)
        for i in range(8):
            advancing.append(third_place_pool[i][0])

        return advancing