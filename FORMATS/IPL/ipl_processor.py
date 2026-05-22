import pandas as pd
import numpy as np
import os
import difflib

print("🔄 Loading IPL database into background memory...")
base_dir = os.path.dirname(__file__)

csv_path = os.path.join(base_dir, "ipl_male_ball_by_ball.csv")

df_ipl = pd.read_csv(csv_path)
print("✅ IPL database loaded and ready.")

def _get_t20_phase(over_num):
    if over_num < 6:
        return "Powerplay (Overs 1-6)"
    elif over_num < 15:
        return "Middle Overs (Overs 7-15)"
    else:
        return "Death Overs (Overs 16-20)"

df_ipl['game_phase'] = df_ipl['over'].apply(_get_t20_phase)

def search_player(search_text):
    all_players = np.unique(np.concatenate([df_ipl['batter'].unique(), df_ipl['bowler'].unique()]))
    search_cleaned = search_text.strip().lower()
    
    substring_matches = [name for name in all_players if search_cleaned in name.lower()]
    
    if substring_matches:
        return sorted(substring_matches)
    
    fuzzy_matches = difflib.get_close_matches(search_text, all_players, n=3, cutoff=0.6)
    
    if not fuzzy_matches:
        return f"❌ No players found matching '{search_text}'."
        
    return fuzzy_matches



def get_ipl_profile(player_name):
    """Filters data, computes metrics, and returns a clean summary table."""

    batter_df = df_ipl[df_ipl['batter'] == player_name].copy()
    
    if batter_df.empty:
        print(f"❌ Player '{player_name}' not found in IPL records.")
        return None
    
    # Isolate legitimate bowler dismissals (Excluding run outs)
    batter_df['true_dismissal'] = (
        (batter_df['player_out'] == player_name) & 
        (~batter_df['wicket_kind'].isin(['run out', 'retired hurt', 'retired out', '']))
    ).astype(int)
    
    
    summary = batter_df.groupby('game_phase').agg(
        Runs=('runs_batter', 'sum'),
        Balls_Faced=('balls_faced', 'sum'),
        Dismissals=('true_dismissal', 'sum')
    ).reset_index()
    
    # Calculate performance rates
    summary['SR'] = ((summary['Runs'] / summary['Balls_Faced']) * 100).round(2)
    summary['Avg'] = np.where(
        summary['Dismissals'] > 0, 
        (summary['Runs'] / summary['Dismissals']).round(2), 
        np.inf
    )
    return summary[['game_phase', 'Runs', 'Balls_Faced', 'Dismissals', 'Avg', 'SR']]
    

def compare_ipl_batters(player1, player2):
    """Fetches profiles for two players and merges them for a clean, phased comparison."""
    # 1. Reuse our existing function to get data for both players
    p1_df = get_ipl_profile(player1)
    p2_df = get_ipl_profile(player2)

    # 2. Safety check in case one of the names is misspelled
    if p1_df is None or p2_df is None:
        print("❌ Comparison aborted due to a missing player profile.")
        return None

    # 3. Insert a 'Player' identifier column at the very front of each table
    p1_df.insert(0, "Player", player1)
    p2_df.insert(0, "Player", player2)

    # 4. Stack the two tables on top of each other
    combined_df = pd.concat([p1_df, p2_df], ignore_index=True)

    # 5. Sort by game_phase first, then by Player name.
    # This ensures Powerplay stats for both players sit right next to each other!
    combined_df = combined_df.sort_values(
        by=["game_phase", "Player"]
    ).reset_index(drop=True)

    return combined_df

    
    
def get_matchup_stats(batter_name, bowler_name):
    """
    Calculates detailed head-to-head statistics between a specific 
    batter and bowler, including runs, boundaries, outs, average, and SR.
    """
    # 1. Filter dataset for this specific face-off
    matchup_df = df_ipl[
        (df_ipl['batter'] == batter_name) & 
        (df_ipl['bowler'] == bowler_name)
    ].copy()
    
    if matchup_df.empty:
        print(f"No recorded data found for {batter_name} vs {bowler_name} in IPL.")
        return None
        
    # 2. Compute aggregate metrics
    runs = matchup_df['runs_batter'].sum()
    balls_faced = matchup_df['balls_faced'].sum()
    fours = (matchup_df['runs_batter'] == 4).sum()
    sixes = (matchup_df['runs_batter'] == 6).sum()
    
    # bowler_wicket is pre-filtered for true bowler dismissals (no run outs!)
    outs = matchup_df['bowler_wicket'].sum()
    
    # 3. Calculate Rate Metrics
    sr = round((runs / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
    avg = round(runs / outs, 2) if outs > 0 else np.inf
    
    # 4. Construct clean display DataFrame
    matchup_data = {
        "Batter": [batter_name],
        "Bowler": [bowler_name],
        "Balls Faced": [balls_faced],
        "Runs": [runs],
        "Fours": [fours],
        "Sixes": [sixes],
        "Outs": [outs],
        "Average": [avg],
        "Strike Rate": [sr]
    }
    
    return pd.DataFrame(matchup_data)



def get_team_toss_stats(team_name):
    """
    Calculates total toss wins and the percentage breakdown 
    of choosing to bat vs field for a specific team.
    """
    # 1. Isolate unique matches to avoid ball-by-ball duplication
    unique_matches = df_ipl.drop_duplicates('match_id')
    
    # 2. Filter for matches where this specific team won the toss
    team_matches_df = unique_matches[(unique_matches['batting_team'] == team_name) | (unique_matches['bowling_team'] == team_name)]
    total_matches = len(team_matches_df)    
    if total_matches == 0:
        print(f"❌ No match records found for '{team_name}'. Ensure exact name.")
        return None
    
    toss_wins_df = team_matches_df[team_matches_df['toss_winner'] == team_name]
    total_toss_wins = len(toss_wins_df)

    toss_wins_pct= round((total_toss_wins / total_matches) * 100, 2)

    toss_losss_pct=round((total_matches-total_toss_wins)/total_matches * 100, 2)
        
    # 3. Aggregate choices
    bat_count = (toss_wins_df['toss_decision'] == 'bat').sum()
    field_count = (toss_wins_df['toss_decision'] == 'field').sum()
    
    # 4. Compute percentages
    bat_pct = round((bat_count / total_toss_wins) * 100, 2)
    field_pct = round((field_count / total_toss_wins) * 100, 2)
    
    toss_summary = {
        "Team": [team_name],
        "Matches Played":[total_matches],
        "Toss Win %":[f"{toss_wins_pct}%"],
        "Toss Loss %":[f"{toss_losss_pct}"],
        "Bat Decision %": [f"{bat_pct}%"],
        "Field Decision %": [f"{field_pct}%"]
    }
    
    return pd.DataFrame(toss_summary)



def get_team_venue_stats(team_name):
    """
    Analyzes a team's win/loss record across IPL grounds.
    Extracts only the core stadium name before the comma to handle duplicates automatically.
    """
    # 1. Isolate unique matches
    unique_matches = df_ipl.drop_duplicates('match_id')
    
    # 2. Filter matches where the team actually played
    team_matches = unique_matches[
        (unique_matches['batting_team'] == team_name) | 
        (unique_matches['bowling_team'] == team_name)
    ].copy()
    
    if team_matches.empty:
        print(f"❌ No match records found for team '{team_name}'.")
        return None
        
    # 3. Create a binary tracker column for a victory
    team_matches['is_win'] = (team_matches['match_won_by'] == team_name).astype(int)

    # team_matches['venue'] = team_matches['venue'].str.replace(r'.*Chinnaswamy.*', 'M Chinnaswamy Stadium', regex=True, case=False)
    # team_matches['venue'] = team_matches['venue'].str.replace(r'.*Chidambaram.*', 'MA Chidambaram Stadium', regex=True, case=False)
    # team_matches['venue'] = team_matches['venue'].str.replace(r'.*Bindra.*', 'IS Bindra Stadium', regex=True, case=False)
    # team_matches['venue'] = team_matches['venue'].str.replace(r'.*Arun Jaitley.*', 'Arun Jaitley Stadium', regex=True, case=False)
    
    # 🚀 THE STRING CLEANER: Extract only the stadium name before the first comma
    team_matches['venue'] = team_matches['venue'].str.split(',').str[0].str.strip()
    
    # 4. Group by the cleaned stadium name and calculate aggregates
    venue_summary = team_matches.groupby('venue').agg(
        Played=('match_id', 'count'),
        Won=('is_win', 'sum')
    ).reset_index()
    
    venue_summary['Lost'] = venue_summary['Played'] - venue_summary['Won']
    venue_summary['Win %'] = ((venue_summary['Won'] / venue_summary['Played']) * 100).round(2)
    
    # 5. Sort by Win % descending, using matches Played ascending as tie-breaker
    final_table = venue_summary.sort_values(
        by=['Win %', 'Played'], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return final_table