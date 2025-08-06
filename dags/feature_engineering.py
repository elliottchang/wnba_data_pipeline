from airflow.decorators import dag, task
from pendulum import datetime, duration
import duckdb, pandas as pd, pathlib
import numpy as np
from typing import Dict, List
from datetime import timedelta

NBA_DB = pathlib.Path("/opt/airflow/data/nba_duck.db")
WNBA_DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 9 * * *",      # run daily at 09:00 UTC (after transformations)
    catchup=False,
    max_active_runs=1,
    tags=["ml", "feature-engineering"]
)
def feature_engineering():
    """Create ML features for predictive models including award predictions and team success forecasting."""

    @task
    def extract_analytics_data(league: str) -> Dict[str, pd.DataFrame]:
        """Extract analytics data for feature engineering."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Extract different data sources
        player_agg = con.execute("SELECT * FROM analytics.player_aggregations").df()
        player_advanced = con.execute("SELECT * FROM analytics.player_advanced_metrics").df()
        player_rolling = con.execute("SELECT * FROM analytics.player_rolling_averages").df()
        
        # Get team data if available
        try:
            team_data = con.execute("SELECT * FROM raw.nba_team_daily" if league == "NBA" else "SELECT * FROM raw.wnba_team_daily").df()
        except:
            team_data = pd.DataFrame()
        
        con.close()
        
        return {
            'player_aggregations': player_agg,
            'player_advanced': player_advanced,
            'player_rolling': player_rolling,
            'team_data': team_data
        }

    @task
    def create_award_prediction_features(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create features for end-of-season award predictions (MVP, DPOY, 6MOY, etc.)."""
        player_agg = data_dict['player_aggregations']
        player_advanced = data_dict['player_advanced']
        player_rolling = data_dict['player_rolling']
        
        if player_agg.empty:
            return pd.DataFrame()
        
        # Merge all player data
        features = player_agg.merge(player_advanced, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_NAME'], how='left')
        
        # MVP Features
        features['mvp_score'] = (
            features['PPG'] * 0.3 +
            features['RPG'] * 0.15 +
            features['APG'] * 0.15 +
            features['SPG'] * 0.1 +
            features['BPG'] * 0.1 +
            features['TRUE_SHOOTING_PCT'] * 100 * 0.2
        )
        
        # DPOY Features
        features['dpoy_score'] = (
            features['SPG'] * 0.3 +
            features['BPG'] * 0.3 +
            features['RPG'] * 0.2 +
            features['TOPG'] * -0.1 +
            features['AVG_PLUS_MINUS'] * 0.1
        )
        
        # 6MOY Features (simplified - would need bench data)
        features['sixmoy_score'] = features['PPG'] * 0.4 + features['APG'] * 0.3 + features['TRUE_SHOOTING_PCT'] * 100 * 0.3
        
        # Team success factor (simplified)
        features['team_success_factor'] = features['AVG_PLUS_MINUS'] * 0.5 + np.random.normal(0, 0.1, len(features))
        
        # Recent performance trend
        if not player_rolling.empty:
            recent_performance = player_rolling.groupby('PLAYER_ID').tail(5)
            recent_avg = recent_performance.groupby('PLAYER_ID')['POINTS_5G'].mean().reset_index()
            recent_avg.columns = ['PLAYER_ID', 'recent_ppg_5g']
            features = features.merge(recent_avg, on='PLAYER_ID', how='left')
            features['performance_trend'] = features['recent_ppg_5g'] - features['PPG']
        else:
            features['performance_trend'] = 0
        
        # Season progress factor
        features['season_progress'] = features['GAME_ID_count'] / 82  # Assuming 82-game season
        
        features['league'] = league
        features['feature_date'] = pd.Timestamp.now()
        
        return features

    @task
    def create_team_success_features(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create features for team success predictions (wins, playoff chances, etc.)."""
        team_data = data_dict['team_data']
        player_agg = data_dict['player_aggregations']
        
        if team_data.empty or player_agg.empty:
            return pd.DataFrame()
        
        team_features = []
        
        for team_name in team_data['team_name'].unique():
            team_players = player_agg[player_agg['TEAM_NAME'] == team_name]
            
            if team_players.empty:
                continue
            
            # Team offensive features
            team_ppg = team_players['PPG'].mean()
            team_apg = team_players['APG'].mean()
            team_fg_pct = team_players['FG_PCT'].mean()
            team_3p_pct = team_players['FG3_PCT'].mean()
            
            # Team defensive features
            team_spg = team_players['SPG'].mean()
            team_bpg = team_players['BPG'].mean()
            team_rpg = team_players['RPG'].mean()
            
            # Team depth features
            team_depth = len(team_players)
            team_star_power = team_players['PPG'].max()  # Best scorer
            
            # Team chemistry (simplified)
            team_chemistry = team_players['APG'].sum() / max(team_players['PPG'].sum(), 1)
            
            # Experience factor
            team_experience = team_players['GAME_ID_count'].mean()
            
            team_features.append({
                'team_name': team_name,
                'team_ppg': team_ppg,
                'team_apg': team_apg,
                'team_fg_pct': team_fg_pct,
                'team_3p_pct': team_3p_pct,
                'team_spg': team_spg,
                'team_bpg': team_bpg,
                'team_rpg': team_rpg,
                'team_depth': team_depth,
                'team_star_power': team_star_power,
                'team_chemistry': team_chemistry,
                'team_experience': team_experience,
                'league': league,
                'feature_date': pd.Timestamp.now()
            })
        
        return pd.DataFrame(team_features)

    @task
    def create_player_performance_features(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create features for individual player performance predictions."""
        player_agg = data_dict['player_aggregations']
        player_rolling = data_dict['player_rolling']
        
        if player_agg.empty:
            return pd.DataFrame()
        
        performance_features = player_agg.copy()
        
        # Efficiency features
        performance_features['efficiency_score'] = (
            performance_features['TRUE_SHOOTING_PCT'] * 0.4 +
            performance_features['FG3_PCT'] * 0.3 +
            performance_features['FT_PCT'] * 0.3
        )
        
        # Usage features
        performance_features['usage_efficiency'] = performance_features['PPG'] / (performance_features['USAGE_RATE'] + 0.01)
        
        # Versatility features
        performance_features['versatility_score'] = (
            performance_features['RPG'] * 0.25 +
            performance_features['APG'] * 0.25 +
            performance_features['SPG'] * 0.25 +
            performance_features['BPG'] * 0.25
        )
        
        # Consistency features (using rolling averages if available)
        if not player_rolling.empty:
            consistency_data = player_rolling.groupby('PLAYER_ID')['POINTS_5G'].std().reset_index()
            consistency_data.columns = ['PLAYER_ID', 'points_consistency']
            performance_features = performance_features.merge(consistency_data, on='PLAYER_ID', how='left')
        else:
            performance_features['points_consistency'] = 0
        
        # Age/experience features (simplified)
        performance_features['experience_factor'] = performance_features['GAME_ID_count'] / 82
        
        # Team context features
        performance_features['team_ppg_rank'] = performance_features.groupby('TEAM_NAME')['PPG'].rank(ascending=False)
        
        performance_features['league'] = league
        performance_features['feature_date'] = pd.Timestamp.now()
        
        return performance_features

    @task
    def create_injury_risk_features(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create features for injury risk prediction."""
        player_agg = data_dict['player_aggregations']
        player_rolling = data_dict['player_rolling']
        
        if player_agg.empty:
            return pd.DataFrame()
        
        injury_features = player_agg.copy()
        
        # Workload features
        injury_features['minutes_per_game'] = injury_features['MINUTES_PLAYED_sum'] / injury_features['GAME_ID_count']
        injury_features['total_workload'] = injury_features['minutes_per_game'] * injury_features['USAGE_RATE']
        
        # Fatigue features
        if not player_rolling.empty:
            recent_games = player_rolling.groupby('PLAYER_ID').tail(10)
            fatigue_score = recent_games.groupby('PLAYER_ID')['MINUTES_PLAYED'].mean().reset_index()
            fatigue_score.columns = ['PLAYER_ID', 'recent_minutes_avg']
            injury_features = injury_features.merge(fatigue_score, on='PLAYER_ID', how='left')
        else:
            injury_features['recent_minutes_avg'] = injury_features['minutes_per_game']
        
        # Physical stress features
        injury_features['physical_stress'] = (
            injury_features['RPG'] * 0.3 +
            injury_features['SPG'] * 0.2 +
            injury_features['BPG'] * 0.2 +
            injury_features['PERSONAL_FOULS_mean'] * 0.3
        )
        
        # Age/experience risk (simplified)
        injury_features['experience_risk'] = injury_features['GAME_ID_count'] / 1000  # Normalized experience
        
        # Injury risk score
        injury_features['injury_risk_score'] = (
            injury_features['total_workload'] * 0.3 +
            injury_features['physical_stress'] * 0.3 +
            injury_features['experience_risk'] * 0.2 +
            injury_features['recent_minutes_avg'] * 0.2
        )
        
        injury_features['league'] = league
        injury_features['feature_date'] = pd.Timestamp.now()
        
        return injury_features

    @task
    def load_ml_features_to_duck(award_features: pd.DataFrame, team_features: pd.DataFrame, 
                                player_features: pd.DataFrame, injury_features: pd.DataFrame, league: str):
        """Load ML features to DuckDB for model training."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Create ML schema
        con.execute("CREATE SCHEMA IF NOT EXISTS ml_features;")
        
        # Load award prediction features
        if not award_features.empty:
            con.execute("DROP TABLE IF EXISTS ml_features.award_prediction_features")
            con.execute("CREATE TABLE ml_features.award_prediction_features AS SELECT * FROM award_features")
        
        # Load team success features
        if not team_features.empty:
            con.execute("DROP TABLE IF EXISTS ml_features.team_success_features")
            con.execute("CREATE TABLE ml_features.team_success_features AS SELECT * FROM team_features")
        
        # Load player performance features
        if not player_features.empty:
            con.execute("DROP TABLE IF EXISTS ml_features.player_performance_features")
            con.execute("CREATE TABLE ml_features.player_performance_features AS SELECT * FROM player_features")
        
        # Load injury risk features
        if not injury_features.empty:
            con.execute("DROP TABLE IF EXISTS ml_features.injury_risk_features")
            con.execute("CREATE TABLE ml_features.injury_risk_features AS SELECT * FROM injury_features")
        
        con.close()

    @task
    def validate_ml_features(award_features: pd.DataFrame, team_features: pd.DataFrame,
                           player_features: pd.DataFrame, injury_features: pd.DataFrame) -> dict:
        """Validate the ML features."""
        validation_results = {
            'award_features_count': len(award_features),
            'team_features_count': len(team_features),
            'player_features_count': len(player_features),
            'injury_features_count': len(injury_features),
            'has_award_features': not award_features.empty,
            'has_team_features': not team_features.empty,
            'has_player_features': not player_features.empty,
            'has_injury_features': not injury_features.empty,
            'validation_passed': True
        }
        
        # Check for required features
        if not award_features.empty:
            required_award_cols = ['mvp_score', 'dpoy_score', 'sixmoy_score']
            missing_cols = [col for col in required_award_cols if col not in award_features.columns]
            if missing_cols:
                validation_results['validation_passed'] = False
                validation_results['error'] = f'Missing award features: {missing_cols}'
        
        return validation_results

    # Process NBA data
    nba_data = extract_analytics_data("NBA")
    nba_award_features = create_award_prediction_features(nba_data, "NBA")
    nba_team_features = create_team_success_features(nba_data, "NBA")
    nba_player_features = create_player_performance_features(nba_data, "NBA")
    nba_injury_features = create_injury_risk_features(nba_data, "NBA")
    load_ml_features_to_duck(nba_award_features, nba_team_features, nba_player_features, nba_injury_features, "NBA")
    
    # Process WNBA data
    wnba_data = extract_analytics_data("WNBA")
    wnba_award_features = create_award_prediction_features(wnba_data, "WNBA")
    wnba_team_features = create_team_success_features(wnba_data, "WNBA")
    wnba_player_features = create_player_performance_features(wnba_data, "WNBA")
    wnba_injury_features = create_injury_risk_features(wnba_data, "WNBA")
    load_ml_features_to_duck(wnba_award_features, wnba_team_features, wnba_player_features, wnba_injury_features, "WNBA")
    
    # Validate features
    validate_ml_features(nba_award_features, nba_team_features, nba_player_features, nba_injury_features)

dag = feature_engineering() 