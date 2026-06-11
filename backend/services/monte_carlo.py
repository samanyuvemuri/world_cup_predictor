import pandas as pd
import numpy as np

class MonteCarloEngine:
    def __init__(self, tournament_groups, schedule, model, feature_builder):
        self.groups = tournament_groups
        self.schedule = schedule
        self.model = model
        self.feature_builder = feature_builder
        self.match_cache = {}
        self.knockout_cache = {}  # NEW: For all possible bracket matchups

    def _build_probability_cache(self):
        """Pre-calculates group stage probabilities and bulletproofs the numpy arrays."""
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
            
            # Bulletproof normalization to prevent np.random.choice ValueError
            prob_array = np.array([probs[0], probs[1], probs[2]], dtype=np.float64)
            prob_array /= prob_array.sum() 
            prob_array[2] = 1.0 - (prob_array[0] + prob_array[1])
            prob_array = np.clip(prob_array, 0.0, 1.0)
            prob_array /= prob_array.sum()
            
            match_id = f"{row['home_team']}_vs_{row['away_team']}"
            self.match_cache[match_id] = prob_array

    def _build_knockout_cache(self):
        """Pre-calculates Win/Loss probabilities for EVERY possible pair of teams (O(1) lookup)."""
        teams = [team for group in self.groups.values() for team in group]
        features_order = ['home_elo', 'away_elo', 'elo_diff', 'is_neutral', 'home_goals_scored_l5', 
                          'home_goals_allowed_l5', 'away_goals_scored_l5', 'away_goals_allowed_l5', 
                          'is_competitive', 'home_rest', 'away_rest', 'rest_diff']
        
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team1, team2 = teams[i], teams[j]
                
                # Knockouts are neutral territory
                features = self.feature_builder.build_features(team1, team2, 1, '2026-07-01')
                X_input = pd.DataFrame([features])[features_order]
                probs = self.model.predict_proba(X_input)[0]
                
                # Normalize to ignore the draw (Force a winner)
                p_team2, p_draw, p_team1 = probs[0], probs[1], probs[2]
                total_p = p_team1 + p_team2
                
                # Save the exact win probability for team1
                normalized_p1 = p_team1 / total_p if total_p > 0 else 0.5
                
                self.knockout_cache[f"{team1}_vs_{team2}"] = normalized_p1
                self.knockout_cache[f"{team2}_vs_{team1}"] = 1.0 - normalized_p1

    def _get_ko_winner(self, team1, team2):
        p1_win = self.knockout_cache[f"{team1}_vs_{team2}"]
        
        if np.random.random() < p1_win:
            return team1
        return team2

    def run_simulation(self, iterations=100000):
        # 1. Pre-calculate all probabilities
        if not self.match_cache:
            self._build_probability_cache()
        if not self.knockout_cache:
            self._build_knockout_cache()

        teams = [team for group in self.groups.values() for team in group]
        
        # 2. Expanded Tracker for all rounds
        results_tracker = {team: {
            "R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Champion": 0
        } for team in teams}

        # 3. The Monte Carlo Loop
        for _ in range(iterations):
            sim_scoreboard = {team: 0 for team in teams}
            
            # --- GROUP STAGE ---
            for index, row in self.schedule.iterrows():
                match_id = f"{row['home_team']}_vs_{row['away_team']}"
                probs = self.match_cache[match_id]
                
                home_points = np.random.choice([0, 1, 3], p=probs)
                
                if home_points == 3:
                    sim_scoreboard[row['home_team']] += 3
                elif home_points == 1:
                    sim_scoreboard[row['home_team']] += 1
                    sim_scoreboard[row['away_team']] += 1
                else:
                    sim_scoreboard[row['away_team']] += 3

            # Determine advancing teams and their group positions (e.g., '1A', '2B', '3C')
            group_stage_results = self._calculate_advancement(sim_scoreboard)
            
            # Log R32 appearances
            for team in group_stage_results.values():
                results_tracker[team]["R32"] += 1

            # --- KNOCKOUT STAGE ---
            knockout_results = self._simulate_bracket(group_stage_results)
            
            # Log advancements for deeper rounds
            for round_name, advancing_teams in knockout_results.items():
                for team in advancing_teams:
                    results_tracker[team][round_name] += 1

        # 4. Convert to Percentages
        final_probs = {}
        for team, stats in results_tracker.items():
            final_probs[team] = {
                k: round((v / iterations) * 100, 2) for k, v in stats.items()
            }

        return final_probs

    def _calculate_advancement(self, sim_scoreboard):
        """Returns a dict mapping FIFA group slot (e.g. '1A', '2B') to the advancing team."""
        group_stage_results = {}
        third_place_pool = []

        for group_name, group_teams in self.groups.items():
            group_standings = sorted([(team, sim_scoreboard[team]) for team in group_teams], 
                                     key=lambda x: x[1], reverse=True)
            
            # Top two advance automatically to fixed slots
            group_stage_results[f"1{group_name}"] = group_standings[0][0]
            group_stage_results[f"2{group_name}"] = group_standings[1][0]
            
            # Save (Points, Group Name, Team Name) for 3rd place sorting
            third_place_pool.append((group_standings[2][1], group_name, group_standings[2][0]))

        # Sort the twelve 3rd-place teams by points and take the top eight
        third_place_pool.sort(key=lambda x: x[0], reverse=True)
        for i in range(8):
            points, group_name, team_name = third_place_pool[i]
            group_stage_results[f"3{group_name}"] = team_name

        return group_stage_results

    def _simulate_bracket(self, group_stage_results):
        """Maps 32 teams to the FIFA routing matrix and plays out the bracket."""
        fixed_teams = {k: v for k, v in group_stage_results.items() if k[0] in ['1', '2']}
        third_place_teams = {k: v for k, v in group_stage_results.items() if k[0] == '3'}
        
        # 1. Allocate the 8 third-place teams (Ensuring they don't play a winner from their own group)
        t_slots = {
            74: {'avoid': 'E'}, 77: {'avoid': 'I'}, 79: {'avoid': 'A'}, 80: {'avoid': 'L'},
            81: {'avoid': 'D'}, 82: {'avoid': 'G'}, 85: {'avoid': 'B'}, 87: {'avoid': 'K'}
        }
        
        allocated_thirds = {}
        unassigned_teams = list(third_place_teams.items()) 
        
        for match_num, rules in t_slots.items():
            for i, (group_pos, team_name) in enumerate(unassigned_teams):
                group_letter = group_pos[1]
                if group_letter != rules['avoid']:
                    allocated_thirds[match_num] = team_name
                    unassigned_teams.pop(i)
                    break

        # 2. Play Round of 32
        matches = {}
        r32 = {
            73: (fixed_teams['2A'], fixed_teams['2B']),
            74: (fixed_teams['1E'], allocated_thirds.get(74, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            75: (fixed_teams['1F'], fixed_teams['2C']),
            76: (fixed_teams['1C'], fixed_teams['2F']),
            77: (fixed_teams['1I'], allocated_thirds.get(77, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            78: (fixed_teams['2E'], fixed_teams['2I']),
            79: (fixed_teams['1A'], allocated_thirds.get(79, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            80: (fixed_teams['1L'], allocated_thirds.get(80, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            81: (fixed_teams['1D'], allocated_thirds.get(81, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            82: (fixed_teams['1G'], allocated_thirds.get(82, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            83: (fixed_teams['2K'], fixed_teams['2L']),
            84: (fixed_teams['1H'], fixed_teams['2J']),
            85: (fixed_teams['1B'], allocated_thirds.get(85, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            86: (fixed_teams['1J'], fixed_teams['2H']),
            87: (fixed_teams['1K'], allocated_thirds.get(87, unassigned_teams[0][1] if unassigned_teams else "Unknown")),
            88: (fixed_teams['2D'], fixed_teams['2G']),
        }

        for match_id, (team1, team2) in r32.items():
            matches[match_id] = self._get_ko_winner(team1, team2)

        # 3. Play Round of 16
        r16 = {
            89: (matches[74], matches[77]), 90: (matches[73], matches[75]),
            91: (matches[76], matches[78]), 92: (matches[79], matches[80]),
            93: (matches[83], matches[84]), 94: (matches[81], matches[82]),
            95: (matches[86], matches[88]), 96: (matches[85], matches[87])
        }
        for match_id, (t1, t2) in r16.items():
            matches[match_id] = self._get_ko_winner(t1, t2)

        # 4. Play Quarterfinals
        qf = {
            97: (matches[89], matches[90]), 98: (matches[93], matches[94]),
            99: (matches[91], matches[92]), 100: (matches[95], matches[96])
        }
        for match_id, (t1, t2) in qf.items():
            matches[match_id] = self._get_ko_winner(t1, t2)

        # 5. Play Semifinals & Final
        matches[101] = self._get_ko_winner(matches[97], matches[98])
        matches[102] = self._get_ko_winner(matches[99], matches[100])
        champion = self._get_ko_winner(matches[101], matches[102])

        # Return the winners of each round
        return {
            "R16": [matches[k] for k in r32.keys()],
            "QF": [matches[k] for k in r16.keys()],
            "SF": [matches[k] for k in qf.keys()],
            "Final": [matches[101], matches[102]],
            "Champion": [champion]
        }