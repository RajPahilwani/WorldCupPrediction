import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Load original file
# ------------------------------------------------------------
df = pd.read_csv('matches_1930_2022 (1).csv')

# ------------------------------------------------------------
# 2. Fix dates using the 'Year' column
# ------------------------------------------------------------
start_dates = {
    2022: '2022/11/20', 2018: '2018/06/14', 2014: '2014/06/12',
    2010: '2010/06/11', 2006: '2006/06/09', 2002: '2002/05/31',
    1998: '1998/06/10', 1994: '1994/06/17', 1990: '1990/06/08',
    1986: '1986/05/31', 1982: '1982/06/13', 1978: '1978/06/01',
    1974: '1974/06/13', 1970: '1970/05/31', 1966: '1966/07/11',
    1962: '1962/05/30', 1958: '1958/06/08', 1954: '1954/06/16',
    1950: '1950/06/24', 1938: '1938/06/04', 1934: '1934/05/27',
    1930: '1930/07/13'
}

def fix_date(row):
    if pd.isna(row['Date']) or row['Date'] == '########':
        return start_dates.get(row.get('Year', 2022), '2022/11/20')
    return row['Date']

df['Date'] = df.apply(fix_date, axis=1)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# ------------------------------------------------------------
# 3. Compute Elo ratings sequentially
# ------------------------------------------------------------
K = 32
elo = {}
for team in pd.concat([df['home_team'], df['away_team']]).unique():
    elo[team] = 1500.0

elo1_list, elo2_list = [], []

for _, row in df.iterrows():
    home, away = row['home_team'], row['away_team']
    g1, g2 = row['home_score'], row['away_score']
    e1, e2 = elo[home], elo[away]
    elo1_list.append(e1); elo2_list.append(e2)

    exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
    exp2 = 1 - exp1
    s1, s2 = (1, 0) if g1 > g2 else ((0, 1) if g1 < g2 else (0.5, 0.5))
    gd = abs(g1 - g2)
    bonus = 1.0 if gd == 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) * 0.25)
    elo[home] += K * bonus * (s1 - exp1)
    elo[away] += K * bonus * (s2 - exp2)

df['Elo1'] = elo1_list
df['Elo2'] = elo2_list

# ------------------------------------------------------------
# 4. Keep only the columns the app expects
# ------------------------------------------------------------
clean = pd.DataFrame({
    'Date': df['Date'],
    'Team 1': df['home_team'],
    'Team 2': df['away_team'],
    'Goals1': df['home_score'],
    'Goals2': df['away_score'],
    'Elo1': df['Elo1'],
    'Elo2': df['Elo2'],
    'Stage': df.get('Round', 'Group').fillna('Group')
})

# ------------------------------------------------------------
# 5. Save the final CSV
# ------------------------------------------------------------
clean.to_csv('world_cup_clean.csv', index=False)
print(f"✅ Done! Created world_cup_clean.csv with {len(clean)} matches.")
print(clean.head())