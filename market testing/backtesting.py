#!/usr/bin/env python3
"""
EV Backtester
=============

Reads a CSV of matches with bookmaker odds (American format) and your
model's predicted probabilities, then for every match:

  1. Converts American odds -> Decimal odds
  2. Computes Expected Value (EV) for home / draw / away using your model's
     probabilities
  3. Picks the highest-EV side (only bets if that EV is > 0)
  4. Settles the bet against the actual result and logs profit/loss
  5. Builds a cumulative PnL curve (bankroll assuming flat 1-unit bets)

Expected CSV columns (header row required):
    home,away,result,home_odds,away_odds,draw_odds,home_predict,away_predict,draw_predict

Column meanings:
    home, away          -> team names (strings)
    result              -> one of: "home", "away", "draw"
                           (also accepts "H"/"A"/"D", or the literal team name)
    home_odds           -> American odds for home win, e.g. -233 or +150
    away_odds           -> American odds for away win
    draw_odds           -> American odds for draw
    home_predict        -> your model's P(home win), e.g. 0.72
    away_predict        -> your model's P(away win)
    draw_predict        -> your model's P(draw)

Usage:
    python ev_backtest.py input.csv --output results.csv --min-edge 0.0

Options:
    --output PATH     Where to write the per-match results CSV
                       (default: <input>_results.csv)
    --min-edge FLOAT  Minimum EV required to place a bet (default: 0.0,
                       i.e. any positive edge). Set higher for a stricter
                       filter, e.g. 0.02 for a 2% minimum edge.
    --unit FLOAT      Size of each flat bet in "units" (default: 1.0)
"""

import argparse
import csv
import sys


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if odds > 0:
        return (odds / 100.0) + 1.0
    elif odds < 0:
        return (100.0 / abs(odds)) + 1.0
    else:
        raise ValueError("American odds cannot be 0")


def normalize_result(result: str, home: str, away: str) -> str:
    """Map a variety of result spellings to 'home' / 'away' / 'draw'."""
    r = result.strip().lower()
    if r in ("home", "h", home.strip().lower()):
        return "home"
    if r in ("away", "a", away.strip().lower()):
        return "away"
    if r in ("draw", "d", "tie"):
        return "draw"
    raise ValueError(f"Could not interpret result value: {result!r}")


def process_match(row: dict, unit: float, min_edge: float) -> dict:
    home = row["home"]
    away = row["away"]

    # --- Step 1: American -> Decimal odds ---
    dec_home = american_to_decimal(row["home_odds"])
    dec_away = american_to_decimal(row["away_odds"])
    dec_draw = american_to_decimal(row["draw_odds"])

    # --- Step 2: Expected Value using model probabilities ---
    p_home = float(row["home_predict"])
    p_away = float(row["away_predict"])
    p_draw = float(row["draw_predict"])

    ev_home = (p_home * dec_home) - 1.0
    ev_away = (p_away * dec_away) - 1.0
    ev_draw = (p_draw * dec_draw) - 1.0

    evs = {"home": ev_home, "away": ev_away, "draw": ev_draw}
    decs = {"home": dec_home, "away": dec_away, "draw": dec_draw}

    # --- Step 3: Decision ---
    best_side = max(evs, key=evs.get)
    best_ev = evs[best_side]

    actual = normalize_result(row["result"], home, away)

    if best_ev > min_edge:
        bet_side = best_side
        won = (bet_side == actual)
        profit = unit * (decs[bet_side] - 1.0) if won else -unit
    else:
        bet_side = None
        won = None
        profit = 0.0

    return {
        "home": home,
        "away": away,
        "result": actual,
        "dec_home": round(dec_home, 3),
        "dec_away": round(dec_away, 3),
        "dec_draw": round(dec_draw, 3),
        "ev_home": round(ev_home, 4),
        "ev_away": round(ev_away, 4),
        "ev_draw": round(ev_draw, 4),
        "bet_side": bet_side if bet_side else "NO BET",
        "won": ("" if won is None else won),
        "profit": round(profit, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="EV Backtester for match odds")
    parser.add_argument("input_csv", nargs="?", default="input.csv",
                        help="Path to the input CSV file (default: input.csv)")
    parser.add_argument("--output", help="Path to write the per-match results CSV", default=None)
    parser.add_argument("--min-edge", type=float, default=0.0,
                         help="Minimum EV edge required to place a bet (default 0.0)")
    parser.add_argument("--unit", type=float, default=1.0,
                         help="Flat bet size in units (default 1.0)")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        if args.input_csv.lower().endswith(".csv"):
            output_path = args.input_csv[:-4] + "_results.csv"
        else:
            output_path = args.input_csv + "_results.csv"

    rows_out = []
    cumulative = 0.0
    bets_placed = 0
    wins = 0
    fieldnames = ["home", "away", "result", "dec_home", "dec_away", "dec_draw",
                  "ev_home", "ev_away", "ev_draw", "bet_side", "won", "profit",
                  "cumulative_pnl"]

    with open(args.input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            with open(output_path, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
            print("Input CSV is empty; no matches processed.")
            print(f"Results written to: {output_path}")
            return

        required_cols = {"home", "away", "result", "home_odds", "away_odds",
                          "draw_odds", "home_predict", "away_predict", "draw_predict"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: input CSV is missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader, start=1):
            try:
                result = process_match(row, unit=args.unit, min_edge=args.min_edge)
            except Exception as e:
                sys.exit(f"ERROR on row {i} ({row.get('home')} vs {row.get('away')}): {e}")

            cumulative += result["profit"]
            result["cumulative_pnl"] = round(cumulative, 4)
            rows_out.append(result)

            if result["bet_side"] != "NO BET":
                bets_placed += 1
                if result["won"] is True:
                    wins += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # --- Summary ---
    total_matches = len(rows_out)
    win_rate = (wins / bets_placed * 100.0) if bets_placed else 0.0
    print(f"Processed {total_matches} matches.")
    print(f"Bets placed: {bets_placed} (min edge = {args.min_edge})")
    print(f"Wins: {wins} / {bets_placed}  ({win_rate:.1f}% hit rate)")
    print(f"Final cumulative PnL: {cumulative:+.3f} units")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()