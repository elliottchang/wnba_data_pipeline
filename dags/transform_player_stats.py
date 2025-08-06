from airflow.decorators import dag, task
from pendulum import datetime, duration
import duckdb, pandas as pd, pathlib
import numpy as np
from typing import Dict, List

NBA_DB = pathlib.Path("/opt/airflow/data/nba_duck.db")
WNBA_DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 8 * * *",      # run daily at 08:00 UTC (after ingestion)
    catchup=False,
    max_active_runs=1,
    tags=["transform", "analytics"]
)
def transform_player_stats():
    """Transform raw player box scores into analytics-ready aggregations and advanced metrics."""

    @task
    def extract_raw_player_data(league: str) -> pd.DataFrame:
        """Extract raw player box score data for the specified league."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        if league == "NBA":
            query = "SELECT * FROM raw.nba_player_box_daily WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 30 DAY"
        else:
            query = "SELECT * FROM raw.wnba_player_box_daily WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 30 DAY"
        
        df = con.execute(query).df()
        con.close()
        return df

    @task
    def calculate_player_aggregations(player_df: pd.DataFrame, league: str) -> pd.DataFrame:
        """Calculate player-level aggregations and averages."""
        if player_df.empty:
            return pd.DataFrame()
        
        # Group by player and calculate aggregations
        agg_stats = player_df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_NAME']).agg({
            'GAME_ID': 'count',  # Games played
            'MINUTES_PLAYED': 'sum',  # Total minutes
            'POINTS': ['sum', 'mean', 'max'],
            'REBOUNDS': ['sum', 'mean', 'max'],
            'ASSISTS': ['sum', 'mean', 'max'],
            'STEALS': ['sum', 'mean', 'max'],
            'BLOCKS': ['sum', 'mean', 'max'],
            'TURNOVERS': ['sum', 'mean'],
            'FIELD_GOALS_MADE': ['sum', 'mean'],
            'FIELD_GOALS_ATTEMPTED': ['sum', 'mean'],
            'THREE_POINTS_MADE': ['sum', 'mean'],
            'THREE_POINTS_ATTEMPTED': ['sum', 'mean'],
            'FREE_THROWS_MADE': ['sum', 'mean'],
            'FREE_THROWS_ATTEMPTED': ['sum', 'mean'],
            'PLUS_MINUS': 'mean'
        }).reset_index()
        
        # Flatten column names
        agg_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_stats.columns]
        
        # Calculate shooting percentages
        agg_stats['FG_PCT'] = (agg_stats['FIELD_GOALS_MADE_sum'] / agg_stats['FIELD_GOALS_ATTEMPTED_sum']).fillna(0)
        agg_stats['FG3_PCT'] = (agg_stats['THREE_POINTS_MADE_sum'] / agg_stats['THREE_POINTS_ATTEMPTED_sum']).fillna(0)
        agg_stats['FT_PCT'] = (agg_stats['FREE_THROWS_MADE_sum'] / agg_stats['FREE_THROWS_ATTEMPTED_sum']).fillna(0)
        
        # Calculate per-game averages
        agg_stats['PPG'] = agg_stats['POINTS_sum'] / agg_stats['GAME_ID_count']
        agg_stats['RPG'] = agg_stats['REBOUNDS_sum'] / agg_stats['GAME_ID_count']
        agg_stats['APG'] = agg_stats['ASSISTS_sum'] / agg_stats['GAME_ID_count']
        agg_stats['SPG'] = agg_stats['STEALS_sum'] / agg_stats['GAME_ID_count']
        agg_stats['BPG'] = agg_stats['BLOCKS_sum'] / agg_stats['GAME_ID_count']
        agg_stats['TOPG'] = agg_stats['TURNOVERS_sum'] / agg_stats['GAME_ID_count']
        
        agg_stats['league'] = league
        agg_stats['last_updated'] = pd.Timestamp.now()
        
        return agg_stats

    @task
    def calculate_advanced_metrics(player_df: pd.DataFrame, league: str) -> pd.DataFrame:
        """Calculate advanced basketball metrics like PER, VORP, etc."""
        if player_df.empty:
            return pd.DataFrame()
        
        # Calculate basic advanced metrics
        advanced_stats = []
        
        for _, player in player_df.groupby(['PLAYER_ID', 'PLAYER_NAME']):
            # Calculate True Shooting Percentage
            total_points = player['POINTS'].sum()
            total_fga = player['FIELD_GOALS_ATTEMPTED'].sum()
            total_fta = player['FREE_THROWS_ATTEMPTED'].sum()
            
            if total_fga + (0.44 * total_fta) > 0:
                ts_pct = total_points / (2 * (total_fga + 0.44 * total_fta))
            else:
                ts_pct = 0
            
            # Calculate Usage Rate (simplified)
            team_possessions = player['GAME_ID'].nunique() * 100  # Simplified
            usage_rate = (player['FIELD_GOALS_ATTEMPTED'].sum() + 0.44 * player['FREE_THROWS_ATTEMPTED'].sum() + player['TURNOVERS'].sum()) / team_possessions
            
            # Calculate Assist Percentage (simplified)
            team_assists = player.groupby('GAME_ID')['ASSISTS'].sum().sum()
            if team_assists > 0:
                ast_pct = player['ASSISTS'].sum() / team_assists
            else:
                ast_pct = 0
            
            advanced_stats.append({
                'PLAYER_ID': player['PLAYER_ID'].iloc[0],
                'PLAYER_NAME': player['PLAYER_NAME'].iloc[0],
                'TEAM_NAME': player['TEAM_NAME'].iloc[0],
                'GAMES_PLAYED': player['GAME_ID'].nunique(),
                'TRUE_SHOOTING_PCT': ts_pct,
                'USAGE_RATE': usage_rate,
                'ASSIST_PCT': ast_pct,
                'AVG_PLUS_MINUS': player['PLUS_MINUS'].mean(),
                'league': league,
                'last_updated': pd.Timestamp.now()
            })
        
        return pd.DataFrame(advanced_stats)

    @task
    def calculate_rolling_averages(player_df: pd.DataFrame, league: str) -> pd.DataFrame:
        """Calculate rolling averages for recent performance trends."""
        if player_df.empty:
            return pd.DataFrame()
        
        # Sort by player and date
        player_df = player_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        rolling_stats = []
        
        for player_id in player_df['PLAYER_ID'].unique():
            player_data = player_df[player_df['PLAYER_ID'] == player_id].copy()
            
            # Calculate 5-game rolling averages
            rolling_5g = player_data.rolling(window=5, min_periods=1).agg({
                'POINTS': 'mean',
                'REBOUNDS': 'mean',
                'ASSISTS': 'mean',
                'STEALS': 'mean',
                'BLOCKS': 'mean',
                'TURNOVERS': 'mean',
                'PLUS_MINUS': 'mean'
            })
            
            # Calculate 10-game rolling averages
            rolling_10g = player_data.rolling(window=10, min_periods=1).agg({
                'POINTS': 'mean',
                'REBOUNDS': 'mean',
                'ASSISTS': 'mean',
                'STEALS': 'mean',
                'BLOCKS': 'mean',
                'TURNOVERS': 'mean',
                'PLUS_MINUS': 'mean'
            })
            
            # Combine with player info
            result = player_data[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_NAME', 'GAME_DATE']].copy()
            result = pd.concat([result, rolling_5g.add_suffix('_5G'), rolling_10g.add_suffix('_10G')], axis=1)
            result['league'] = league
            result['last_updated'] = pd.Timestamp.now()
            
            rolling_stats.append(result)
        
        if rolling_stats:
            return pd.concat(rolling_stats, ignore_index=True)
        return pd.DataFrame()

    @task
    def load_analytics_data_to_duck(agg_df: pd.DataFrame, advanced_df: pd.DataFrame, rolling_df: pd.DataFrame, league: str):
        """Load transformed analytics data to DuckDB."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Create analytics schema
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
        
        # Load player aggregations
        if not agg_df.empty:
            con.execute("DROP TABLE IF EXISTS analytics.player_aggregations")
            con.execute("CREATE TABLE analytics.player_aggregations AS SELECT * FROM agg_df")
        
        # Load advanced metrics
        if not advanced_df.empty:
            con.execute("DROP TABLE IF EXISTS analytics.player_advanced_metrics")
            con.execute("CREATE TABLE analytics.player_advanced_metrics AS SELECT * FROM advanced_df")
        
        # Load rolling averages
        if not rolling_df.empty:
            con.execute("DROP TABLE IF EXISTS analytics.player_rolling_averages")
            con.execute("CREATE TABLE analytics.player_rolling_averages AS SELECT * FROM rolling_df")
        
        con.close()

    @task
    def validate_transformations(agg_df: pd.DataFrame, advanced_df: pd.DataFrame, rolling_df: pd.DataFrame) -> dict:
        """Validate the transformed data."""
        validation_results = {
            'aggregations_count': len(agg_df),
            'advanced_metrics_count': len(advanced_df),
            'rolling_averages_count': len(rolling_df),
            'has_aggregations': not agg_df.empty,
            'has_advanced_metrics': not advanced_df.empty,
            'has_rolling_averages': not rolling_df.empty,
            'validation_passed': True
        }
        
        # Check for data quality issues
        if not agg_df.empty:
            if agg_df['PPG'].isnull().any():
                validation_results['validation_passed'] = False
                validation_results['error'] = 'Missing PPG calculations'
        
        return validation_results

    # Process NBA data
    nba_raw = extract_raw_player_data("NBA")
    nba_agg = calculate_player_aggregations(nba_raw, "NBA")
    nba_advanced = calculate_advanced_metrics(nba_raw, "NBA")
    nba_rolling = calculate_rolling_averages(nba_raw, "NBA")
    load_analytics_data_to_duck(nba_agg, nba_advanced, nba_rolling, "NBA")
    
    # Process WNBA data
    wnba_raw = extract_raw_player_data("WNBA")
    wnba_agg = calculate_player_aggregations(wnba_raw, "WNBA")
    wnba_advanced = calculate_advanced_metrics(wnba_raw, "WNBA")
    wnba_rolling = calculate_rolling_averages(wnba_raw, "WNBA")
    load_analytics_data_to_duck(wnba_agg, wnba_advanced, wnba_rolling, "WNBA")
    
    # Validate transformations
    validate_transformations(nba_agg, nba_advanced, nba_rolling)

dag = transform_player_stats() 