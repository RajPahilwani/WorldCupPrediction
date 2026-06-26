"""
World Cup Player Performance Scraper
Uses cloudscraper for all HTTP requests (GitHub CSVs + FBref).
No external package dependencies beyond cloudscraper and pandas.
"""

import pandas as pd
import cloudscraper
import time
import os
from io import StringIO
from typing import Optional

# ============================================================
# CONSTANTS
# ============================================================

GITHUB_BASE = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/"
FILES = {
    'appearances': 'appearances.csv',
    'matches': 'matches.csv',
    'players': 'players.csv'
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ============================================================
# HELPER: Fetch CSV with cloudscraper (with retry)
# ============================================================

def fetch_csv(url: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Downloads a CSV from a URL using cloudscraper with retries.
    Returns DataFrame or None if failed.
    """
    scraper = cloudscraper.create_scraper()
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(retries):
        try:
            print(f"  Fetching {url} (attempt {attempt+1})...")
            response = scraper.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Check if it's actually CSV (not HTML)
            if 'text/csv' in response.headers.get('Content-Type', '') or url.endswith('.csv'):
                df = pd.read_csv(StringIO(response.text))
                return df
            else:
                print(f"  ⚠ Unexpected content type, trying to parse anyway...")
                df = pd.read_csv(StringIO(response.text))
                return df
        except Exception as e:
            print(f"  ✗ Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None

# ============================================================
# LOAD HISTORICAL DATA FROM GITHUB (via cloudscraper)
# ============================================================

def load_historical_data() -> tuple:
    """
    Loads appearances, matches, players from GitHub CSV using cloudscraper.
    Returns (appearances_df, matches_df, players_df) or (None, None, None).
    """
    print("Loading historical data from GitHub (via cloudscraper)...")
    data = {}

    for name, filename in FILES.items():
        url = GITHUB_BASE + filename
        df = fetch_csv(url)
        if df is not None and not df.empty:
            data[name] = df
            print(f"  ✓ Loaded {len(df)} {name}")
        else:
            print(f"  ✗ Failed to load {name}")
            return None, None, None

    return data.get('appearances'), data.get('matches'), data.get('players')

# ============================================================
# FALLBACK: Load from local files if available
# ============================================================

def load_local_csv(filename: str) -> Optional[pd.DataFrame]:
    """Load a CSV from the current directory if it exists."""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            print(f"  ✓ Loaded {len(df)} from local {filename}")
            return df
        except Exception as e:
            print(f"  ✗ Error loading local {filename}: {e}")
    return None

# ============================================================
# FBref SCRAPER (with cloudscraper)
# ============================================================

def scrape_fbref_world_cup(year: int) -> Optional[pd.DataFrame]:
    if year == 2026:
        url = "https://fbref.com/en/comps/1/stats/World-Cup-Stats"
    else:
        url = f"https://fbref.com/en/comps/1/stats/{year}-World-Cup-Stats"

    headers = {"User-Agent": USER_AGENT}
    print(f"Scraping FBref for {year}...")

    scraper = cloudscraper.create_scraper()

    try:
        response = scraper.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            # Look for the standard stats table (contains 'Player' and many numeric columns)
            if len(table.columns) > 10 and 'Player' in str(table.columns[0]):
                df = table
                df['world_cup_year'] = year
                # Clean column names
                df.columns = [col.replace(' ', '_').lower() for col in df.columns]
                # Remove summary rows
                if 'player' in df.columns:
                    df = df[~df['player'].str.contains('Squad Total|Team Total', na=False)]
                print(f"  ✓ {len(df)} players found for {year}")
                return df

        print(f"  ⚠ No stats table found for {year}")
        return None
    except Exception as e:
        print(f"  ✗ Error scraping {year}: {e}")
        return None

# ============================================================
# BUILD DATASETS
# ============================================================

def build_complete_dataset():
    print("=" * 60)
    print("BUILDING WORLD CUP PLAYER PERFORMANCE DATASET")
    print("(using cloudscraper for all HTTP requests)")
    print("=" * 60)

    datasets = {}

    # 1. Historical data
    appearances, matches, players = load_historical_data()

    # If GitHub fails, try local files
    if appearances is None:
        print("Attempting to load from local CSV files...")
        appearances = load_local_csv("appearances.csv")
        matches = load_local_csv("matches.csv")
        players = load_local_csv("players.csv")

    if appearances is not None:
        datasets['historical_appearances'] = appearances
        datasets['historical_matches'] = matches
        datasets['historical_players'] = players
    else:
        print("⚠ Could not load historical data. Please download manually from:")
        print("  https://github.com/jfjelstul/worldcup/tree/master/data-csv")
        print("  and place appearances.csv, matches.csv, players.csv in this folder.")
        # Continue anyway to scrape FBref and 2026 APIs

    # 2. FBref detailed stats
    fbref_list = []
    for year in [2026, 2022, 2018, 2014, 2010]:
        df = scrape_fbref_world_cup(year)
        if df is not None and not df.empty:
            fbref_list.append(df)
        time.sleep(2)  # Rate limiting

    if fbref_list:
        datasets['fbref_stats'] = pd.concat(fbref_list, ignore_index=True)

    # 3. 2026 API (free, no key)
    try:
        import requests
        print("Fetching 2026 match data from free API...")
        resp = requests.get("https://worldcup2026-api.vercel.app/api/matches", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            datasets['api_2026'] = df
            print(f"  ✓ Loaded {len(df)} matches from API")
    except Exception as e:
        print(f"  ✗ Could not fetch 2026 API: {e}")

    # 4. ESPN public API (optional)
    try:
        print("Fetching 2026 data from ESPN public API...")
        resp = requests.get("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.worldcup/scoreboard", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            matches = []
            for event in data.get('events', []):
                comp = event.get('competitions', [{}])[0]
                competitors = comp.get('competitors', [])
                if len(competitors) >= 2:
                    matches.append({
                        'match_id': event.get('id'),
                        'date': event.get('date'),
                        'status': event.get('status', {}).get('type', {}).get('description'),
                        'team1': competitors[0].get('team', {}).get('displayName'),
                        'team2': competitors[1].get('team', {}).get('displayName'),
                        'score1': competitors[0].get('score'),
                        'score2': competitors[1].get('score'),
                    })
            df = pd.DataFrame(matches)
            datasets['espn_2026'] = df
            print(f"  ✓ Loaded {len(df)} matches from ESPN")
    except Exception as e:
        print(f"  ✗ ESPN API error: {e}")

    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    for key, df in datasets.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {key}: {len(df)} rows")
    print("=" * 60)

    return datasets

# ============================================================
# SAVE
# ============================================================

def save_datasets(datasets):
    for name, df in datasets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            filename = f"world_cup_{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    datasets = build_complete_dataset()
    save_datasets(datasets)