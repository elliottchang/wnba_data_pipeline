from airflow.decorators import dag, task
from pendulum import datetime, duration
import requests, duckdb, pandas as pd, pathlib
from nba_api.stats.endpoints import boxscoretraditionalv2
import logging

DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 6 * * *",      # run daily at 06:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["wnba", "daily"]
)
def ingest_wnba_daily():
    """Pull WNBA player box scores and team data for games played on the execution date."""

    @task(retries=3, retry_delay=duration(minutes=5))
    def get_wnba_game_ids(date_str: str) -> list[str]:
        """Discover WNBA game IDs using ESPN's public scoreboard feed."""
        yyyymmdd = date_str.replace("-", "")
        url = (
            "https://site.api.espn.com/apis/v2/sports/"
            f"basketball/wnba/scoreboard?dates={yyyymmdd}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [event["id"] for event in data.get("events", [])]
        except requests.HTTPError as err:
            if err.response.status_code == 404:
                logging.info(f"No WNBA games on {date_str}")
                return []
            raise

    @task
    def fetch_wnba_player_box(game_ids: list[str]) -> pd.DataFrame:
        """Fetch WNBA player box scores from stats.nba.com via nba_api."""
        if not game_ids:
            return pd.DataFrame()
        
        dfs = []
        for game_id in game_ids:
            try:
                box_data = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                df = box_data.get_data_frames()[0]
                df['game_id'] = game_id
                df['league'] = 'WNBA'
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error fetching box score for game {game_id}: {e}")
                continue
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    @task
    def fetch_wnba_team_stats(date_str: str) -> pd.DataFrame:
        """Fetch WNBA team standings and basic stats."""
        # This would typically use a WNBA-specific API
        # For now, we'll create a placeholder structure
        teams = [
            'Atlanta Dream', 'Chicago Sky', 'Connecticut Sun', 'Dallas Wings',
            'Indiana Fever', 'Las Vegas Aces', 'Los Angeles Sparks', 'Minnesota Lynx',
            'New York Liberty', 'Phoenix Mercury', 'Seattle Storm', 'Washington Mystics'
        ]
        
        # Placeholder data - in production, this would come from an API
        team_data = []
        for team in teams:
            team_data.append({
                'team_name': team,
                'date': date_str,
                'league': 'WNBA',
                'games_played': 0,  # Would be populated from API
                'wins': 0,
                'losses': 0,
                'win_percentage': 0.0
            })
        
        return pd.DataFrame(team_data)

    @task
    def load_wnba_player_data_to_duck(player_df: pd.DataFrame):
        """Load WNBA player box scores to DuckDB."""
        con = duckdb.connect(DB)
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.wnba_player_box_daily (
                PLAYER_ID INTEGER,
                PLAYER_NAME VARCHAR,
                TEAM_ID INTEGER,
                TEAM_NAME VARCHAR,
                GAME_ID VARCHAR,
                GAME_DATE DATE,
                MINUTES_PLAYED VARCHAR,
                FIELD_GOALS_MADE INTEGER,
                FIELD_GOALS_ATTEMPTED INTEGER,
                FIELD_GOAL_PERCENTAGE DECIMAL,
                THREE_POINTS_MADE INTEGER,
                THREE_POINTS_ATTEMPTED INTEGER,
                THREE_POINT_PERCENTAGE DECIMAL,
                FREE_THROWS_MADE INTEGER,
                FREE_THROWS_ATTEMPTED INTEGER,
                FREE_THROW_PERCENTAGE DECIMAL,
                OFFENSIVE_REBOUNDS INTEGER,
                DEFENSIVE_REBOUNDS INTEGER,
                REBOUNDS INTEGER,
                ASSISTS INTEGER,
                STEALS INTEGER,
                BLOCKS INTEGER,
                TURNOVERS INTEGER,
                PERSONAL_FOULS INTEGER,
                POINTS INTEGER,
                PLUS_MINUS INTEGER,
                league VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        if not player_df.empty:
            con.execute("INSERT INTO raw.wnba_player_box_daily SELECT * FROM player_df")
        
        con.close()

    @task
    def load_wnba_team_data_to_duck(team_df: pd.DataFrame):
        """Load WNBA team data to DuckDB."""
        con = duckdb.connect(DB)
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS raw.wnba_team_daily (
                team_name VARCHAR,
                date DATE,
                league VARCHAR,
                games_played INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_percentage DECIMAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        if not team_df.empty:
            con.execute("INSERT INTO raw.wnba_team_daily SELECT * FROM team_df")
        
        con.close()

    @task
    def validate_wnba_data(player_df: pd.DataFrame, team_df: pd.DataFrame) -> dict:
        """Validate the ingested WNBA data."""
        validation_results = {
            'player_records_count': len(player_df),
            'team_records_count': len(team_df),
            'has_player_data': not player_df.empty,
            'has_team_data': not team_df.empty,
            'validation_passed': True
        }
        
        # Basic validation checks
        if player_df.empty and team_df.empty:
            validation_results['validation_passed'] = False
            validation_results['error'] = 'No data ingested'
        
        logging.info(f"WNBA data validation: {validation_results}")
        return validation_results

    # Task dependencies
    game_ids = get_wnba_game_ids("{{ ds }}")
    player_box = fetch_wnba_player_box(game_ids)
    team_stats = fetch_wnba_team_stats("{{ ds }}")
    
    # Load data to database
    load_wnba_player_data_to_duck(player_box)
    load_wnba_team_data_to_duck(team_stats)
    
    # Validate data
    validate_wnba_data(player_box, team_stats)

dag = ingest_wnba_daily() 