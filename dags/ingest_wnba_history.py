from airflow.decorators import dag, task
from pendulum import datetime, duration
import requests, duckdb, pandas as pd, pathlib
from nba_api.stats.endpoints import boxscoretraditionalv2
import logging
import time
from typing import List

DB = pathlib.Path("/opt/airflow/data/nba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 6 * * *",      # run daily at 06:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["nba", "daily"]
)
def ingest_nba_daily():
    """Pull NBA player box scores for games played on the execution date."""

    # Discover game_ids using ESPN's public scoreboard feed
    @task(retries=3, retry_delay=duration(minutes=5))
    def get_game_ids(date_str: str) -> list[str]:
        yyyymmdd = date_str.replace("-", "")
        url = (
            "https://site.api.espn.com/apis/v2/sports/"
            f"basketball/nba/scoreboard?dates={yyyymmdd}"
        )
        try:
            logging.info(f"Fetching NBA games for date: {date_str}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            game_ids = [event["id"] for event in data.get("events", [])]
            logging.info(f"Found {len(game_ids)} NBA games for {date_str}")
            return game_ids
        except requests.HTTPError as err:
            if err.response.status_code == 404:
                logging.info(f"No NBA games found on {date_str}")
                return []
            logging.error(f"HTTP error fetching NBA games: {err}")
            raise
        except Exception as e:
            logging.error(f"Error fetching NBA games: {e}")
            raise

    # Fetch per‑player box scores from stats.nba.com via nba_api
    @task(retries=2, retry_delay=duration(minutes=2))
    def fetch_player_box(game_ids: list[str]) -> pd.DataFrame:
        if not game_ids:
            logging.info("No game IDs provided, returning empty DataFrame")
            return pd.DataFrame()
        
        all_dfs = []
        successful_games = 0
        failed_games = 0
        
        for game_id in game_ids:
            try:
                logging.info(f"Fetching box score for NBA game: {game_id}")
                
                # Add delay to respect rate limits
                time.sleep(1)
                
                # Fetch box score data
                box_data = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                df = box_data.get_data_frames()[0]
                
                if not df.empty:
                    # Add metadata
                    df['game_id'] = game_id
                    df['league'] = 'NBA'
                    df['created_at'] = pd.Timestamp.now()
                    
                    all_dfs.append(df)
                    successful_games += 1
                    logging.info(f"Successfully fetched box score for game {game_id} with {len(df)} player records")
                else:
                    logging.warning(f"Empty box score data for game {game_id}")
                    failed_games += 1
                    
            except Exception as e:
                logging.error(f"Error fetching box score for game {game_id}: {e}")
                failed_games += 1
                continue
        
        if all_dfs:
            result_df = pd.concat(all_dfs, ignore_index=True)
            logging.info(f"Successfully processed {successful_games} games, {failed_games} failed. Total records: {len(result_df)}")
            return result_df
        else:
            logging.warning("No successful box score fetches")
            return pd.DataFrame()

    # Load/append to DuckDB
    @task
    def load_to_duck(df: pd.DataFrame):
        if df.empty:
            logging.info("No data to load to DuckDB")
            return
        
        try:
            con = duckdb.connect(DB)
            con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
            
            # Create table with proper schema
            con.execute("""
                CREATE TABLE IF NOT EXISTS raw.nba_player_box_daily (
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
            """)
            
            # Insert data
            con.execute("INSERT INTO raw.nba_player_box_daily SELECT * FROM df")
            con.close()
            
            logging.info(f"Successfully loaded {len(df)} records to DuckDB")
            
        except Exception as e:
            logging.error(f"Error loading data to DuckDB: {e}")
            raise

    # Validate the ingested data
    @task
    def validate_ingestion(df: pd.DataFrame) -> dict:
        validation_results = {
            'records_ingested': len(df),
            'unique_players': df['PLAYER_NAME'].nunique() if not df.empty else 0,
            'unique_teams': df['TEAM_NAME'].nunique() if not df.empty else 0,
            'unique_games': df['GAME_ID'].nunique() if not df.empty else 0,
            'validation_passed': True
        }
        
        if df.empty:
            validation_results['validation_passed'] = False
            validation_results['warning'] = 'No data ingested'
        elif len(df) < 10:  # Expect at least 10 player records
            validation_results['validation_passed'] = False
            validation_results['warning'] = f'Only {len(df)} records ingested, expected more'
        
        logging.info(f"Validation results: {validation_results}")
        return validation_results

    # Task dependencies
    ids = get_game_ids("{{ ds }}")
    box = fetch_player_box(ids)
    load_to_duck(box)
    validate_ingestion(box)

dag = ingest_nba_daily()
