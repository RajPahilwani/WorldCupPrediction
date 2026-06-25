import pandas as pd
import ast
from datetime import datetime

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------
current_df = pd.read_csv('world_cup_clean.csv')   # your simple dataset
complex_df = pd.read_csv('matches_World_Cup.csv')  # the detailed CSV

# ------------------------------------------------------------
# 2. Helper functions to parse player stats from formation columns
# ------------------------------------------------------------

def safe_literal_eval(val):
    """Safely evaluate a string containing a Python literal."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (SyntaxError, ValueError):
            return None
    return val

def extract_player_stats(lineup_str, subs_str):
    """
    Given string representations of 'lineup' and 'substitutions',
    return a dictionary {player_id: {'goals': total, 'assists': total}}.
    """
    lineup = safe_literal_eval(lineup_str) or []
    subs = safe_literal_eval(subs_str) or []
    all_players = lineup + subs

    stats = {}
    for player in all_players:
        if not isinstance(player, dict):
            continue
        pid = player.get('playerId')
        if not pid:
            continue
        # goals
        g = player.get('goals')
        if g is not None and g != 'null':
            try:
                g = int(g)
            except (ValueError, TypeError):
                g = 0
        else:
            g = 0
        # assists
        a = player.get('assists')
        if a is not None and a != 'null':
            try:
                a = int(a)
            except (ValueError, TypeError):
                a = 0
        else:
            a = 0

        if pid not in stats:
            stats[pid] = {'goals': 0, 'assists': 0}
        stats[pid]['goals'] += g
        stats[pid]['assists'] += a

    return stats

def get_top_scorer_and_assister(stats_dict):
    """
    From a stats_dict {pid: {'goals': int, 'assists': int}},
    return:
        (top_scorer_id, top_scorer_goals, top_assist_id, top_assist_assists)
    If ties, pick the first encountered.
    """
    if not stats_dict:
        return None, 0, None, 0

    # Top goals
    best_g = max(stats_dict.items(), key=lambda x: x[1]['goals'])
    top_scorer_id, top_scorer_goals = best_g[0], best_g[1]['goals']

    # Top assists
    best_a = max(stats_dict.items(), key=lambda x: x[1]['assists'])
    top_assist_id, top_assist_assists = best_a[0], best_a[1]['assists']

    return top_scorer_id, top_scorer_goals, top_assist_id, top_assist_assists

def parse_complex_row(row):
    """
    Extract match data and top stats for both teams.
    """
    # Date: take only YYYY-MM-DD
    date_str = row['dateutc']
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        date = dt.strftime('%Y-%m-%d')
    except:
        date = date_str.split()[0]  # fallback

    # Team names from label: "Home - Away, score"
    label = row['label']
    parts = label.split(' - ')
    if len(parts) >= 2:
        home_team = parts[0].strip()
        away_part = parts[1].split(',')[0].strip()
        away_team = away_part
    else:
        home_team = away_team = None

    home_goals = int(row['team1.score']) if pd.notna(row['team1.score']) else 0
    away_goals = int(row['team2.score']) if pd.notna(row['team2.score']) else 0

    # ---- Team 1 ----
    lineup1 = row.get('team1.formation.lineup', '[]')
    subs1 = row.get('team1.formation.substitutions', '[]')
    stats1 = extract_player_stats(lineup1, subs1)
    (scorer1_id, scorer1_goals, assist1_id, assist1_assists) = get_top_scorer_and_assister(stats1)

    # ---- Team 2 ----
    lineup2 = row.get('team2.formation.lineup', '[]')
    subs2 = row.get('team2.formation.substitutions', '[]')
    stats2 = extract_player_stats(lineup2, subs2)
    (scorer2_id, scorer2_goals, assist2_id, assist2_assists) = get_top_scorer_and_assister(stats2)

    return {
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'top_scorer1_id': scorer1_id,
        'top_scorer1_goals': scorer1_goals,
        'top_assist1_id': assist1_id,
        'top_assist1_assists': assist1_assists,
        'top_scorer2_id': scorer2_id,
        'top_scorer2_goals': scorer2_goals,
        'top_assist2_id': assist2_id,
        'top_assist2_assists': assist2_assists,
    }

# ------------------------------------------------------------
# 3. Parse the complex CSV row by row
# ------------------------------------------------------------
parsed_rows = []
for idx, row in complex_df.iterrows():
    try:
        parsed_rows.append(parse_complex_row(row))
    except Exception as e:
        print(f"Warning: row {idx} failed: {e}")
        continue

parsed_df = pd.DataFrame(parsed_rows)
parsed_df = parsed_df.dropna(subset=['home_team', 'away_team'])
parsed_df = parsed_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])

# Rename columns to match current dataset for merging
parsed_df = parsed_df.rename(columns={
    'date': 'Date',
    'home_team': 'Team 1',
    'away_team': 'Team 2',
    'home_goals': 'Goals1',
    'away_goals': 'Goals2',
})

# Columns to add (exclude Date, Team1, Team2, Goals1, Goals2)
add_cols = [
    'top_scorer1_id', 'top_scorer1_goals', 'top_assist1_id', 'top_assist1_assists',
    'top_scorer2_id', 'top_scorer2_goals', 'top_assist2_id', 'top_assist2_assists'
]

# ------------------------------------------------------------
# 4. Merge with current dataset
# ------------------------------------------------------------
merged = current_df.merge(
    parsed_df[['Date', 'Team 1', 'Team 2'] + add_cols],
    on=['Date', 'Team 1', 'Team 2'],
    how='left'
)

# ------------------------------------------------------------
# 5. Save result
# ------------------------------------------------------------
merged.to_csv('merged_worldcup_with_stats.csv', index=False)

print(f"Merged successfully. Output: merged_worldcup_with_stats.csv")
print(f"Rows in original: {len(current_df)}, after merge: {len(merged)}")
print(f"Added columns: {add_cols}")