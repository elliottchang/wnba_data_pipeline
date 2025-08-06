from airflow.decorators import dag, task
from pendulum import datetime, duration
import duckdb, pandas as pd, pathlib
import numpy as np
import json
import pickle
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging

NBA_DB = pathlib.Path("/opt/airflow/data/nba_duck.db")
WNBA_DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")
MODEL_DIR = pathlib.Path("/opt/airflow/models")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 12 * * 0",      # run weekly on Sundays at 12:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["ml", "model-training"]
)
def model_training():
    """Train ML models for award predictions, team success forecasting, and player performance predictions."""

    @task
    def extract_training_data(league: str) -> Dict[str, pd.DataFrame]:
        """Extract ML features for model training."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Extract ML features
        try:
            award_features = con.execute("SELECT * FROM ml_features.award_prediction_features").df()
            team_features = con.execute("SELECT * FROM ml_features.team_success_features").df()
            player_features = con.execute("SELECT * FROM ml_features.player_performance_features").df()
            injury_features = con.execute("SELECT * FROM ml_features.injury_risk_features").df()
        except Exception as e:
            logging.warning(f"Could not extract ML features for {league}: {e}")
            award_features = pd.DataFrame()
            team_features = pd.DataFrame()
            player_features = pd.DataFrame()
            injury_features = pd.DataFrame()
        
        con.close()
        
        return {
            'award_features': award_features,
            'team_features': team_features,
            'player_features': player_features,
            'injury_features': injury_features
        }

    @task
    def train_award_prediction_model(award_features: pd.DataFrame, league: str) -> Dict[str, any]:
        """Train model for award predictions (MVP, DPOY, 6MOY)."""
        if award_features.empty:
            return {'model_trained': False, 'error': 'No award features available'}
        
        try:
            # Prepare features for MVP prediction
            feature_cols = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 
                          'TRUE_SHOOTING_PCT', 'USAGE_RATE', 'ASSIST_PCT', 'AVG_PLUS_MINUS']
            
            # Use MVP score as target (simplified approach)
            X = award_features[feature_cols].fillna(0)
            y = award_features['mvp_score'].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # Save model
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / f"{league.lower()}_award_prediction_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            scaler_path = MODEL_DIR / f"{league.lower()}_award_prediction_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            results = {
                'model_trained': True,
                'league': league,
                'model_type': 'award_prediction',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logging.info(f"Award prediction model trained for {league}: R² = {r2:.3f}, MSE = {mse:.3f}")
            return results
            
        except Exception as e:
            logging.error(f"Error training award prediction model for {league}: {e}")
            return {'model_trained': False, 'error': str(e)}

    @task
    def train_team_success_model(team_features: pd.DataFrame, league: str) -> Dict[str, any]:
        """Train model for team success predictions."""
        if team_features.empty:
            return {'model_trained': False, 'error': 'No team features available'}
        
        try:
            # Prepare features for team success prediction
            feature_cols = ['team_ppg', 'team_apg', 'team_fg_pct', 'team_3p_pct', 
                          'team_spg', 'team_bpg', 'team_rpg', 'team_depth', 
                          'team_star_power', 'team_chemistry', 'team_experience']
            
            # Create synthetic target (win percentage) - in real scenario, this would be actual team records
            X = team_features[feature_cols].fillna(0)
            y = (team_features['team_ppg'] + team_features['team_apg']) / 2  # Simplified target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # Save model
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / f"{league.lower()}_team_success_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'model_trained': True,
                'league': league,
                'model_type': 'team_success',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logging.info(f"Team success model trained for {league}: R² = {r2:.3f}, MSE = {mse:.3f}")
            return results
            
        except Exception as e:
            logging.error(f"Error training team success model for {league}: {e}")
            return {'model_trained': False, 'error': str(e)}

    @task
    def train_player_performance_model(player_features: pd.DataFrame, league: str) -> Dict[str, any]:
        """Train model for player performance predictions."""
        if player_features.empty:
            return {'model_trained': False, 'error': 'No player features available'}
        
        try:
            # Prepare features for player performance prediction
            feature_cols = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                          'TRUE_SHOOTING_PCT', 'USAGE_RATE', 'ASSIST_PCT', 'AVG_PLUS_MINUS',
                          'efficiency_score', 'usage_efficiency', 'versatility_score', 'experience_factor']
            
            # Use overall rating as target
            X = player_features[feature_cols].fillna(0)
            y = player_features['overall_rating'].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # Save model
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / f"{league.lower()}_player_performance_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'model_trained': True,
                'league': league,
                'model_type': 'player_performance',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logging.info(f"Player performance model trained for {league}: R² = {r2:.3f}, MSE = {mse:.3f}")
            return results
            
        except Exception as e:
            logging.error(f"Error training player performance model for {league}: {e}")
            return {'model_trained': False, 'error': str(e)}

    @task
    def train_injury_risk_model(injury_features: pd.DataFrame, league: str) -> Dict[str, any]:
        """Train model for injury risk predictions."""
        if injury_features.empty:
            return {'model_trained': False, 'error': 'No injury features available'}
        
        try:
            # Prepare features for injury risk prediction
            feature_cols = ['minutes_per_game', 'total_workload', 'recent_minutes_avg',
                          'physical_stress', 'experience_risk', 'USAGE_RATE', 'GAME_ID_count']
            
            # Create synthetic injury risk target (in real scenario, this would be actual injury data)
            X = injury_features[feature_cols].fillna(0)
            y = (injury_features['total_workload'] > injury_features['total_workload'].median()).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # Save model
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / f"{league.lower()}_injury_risk_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'model_trained': True,
                'league': league,
                'model_type': 'injury_risk',
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logging.info(f"Injury risk model trained for {league}: Accuracy = {accuracy:.3f}")
            return results
            
        except Exception as e:
            logging.error(f"Error training injury risk model for {league}: {e}")
            return {'model_trained': False, 'error': str(e)}

    @task
    def save_model_metadata(award_results: Dict, team_results: Dict, 
                           player_results: Dict, injury_results: Dict, league: str):
        """Save model metadata and performance metrics."""
        
        metadata = {
            'league': league,
            'training_date': datetime.utcnow().isoformat(),
            'models': {
                'award_prediction': award_results,
                'team_success': team_results,
                'player_performance': player_results,
                'injury_risk': injury_results
            },
            'summary': {
                'total_models_trained': sum([
                    award_results.get('model_trained', False),
                    team_results.get('model_trained', False),
                    player_results.get('model_trained', False),
                    injury_results.get('model_trained', False)
                ]),
                'successful_models': [
                    model_type for model_type, results in [
                        ('award_prediction', award_results),
                        ('team_success', team_results),
                        ('player_performance', player_results),
                        ('injury_risk', injury_results)
                    ] if results.get('model_trained', False)
                ]
            }
        }
        
        # Save metadata
        MODEL_DIR.mkdir(exist_ok=True)
        metadata_path = MODEL_DIR / f"{league.lower()}_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Model metadata saved for {league}: {metadata['summary']['total_models_trained']} models trained")

    # Train NBA models
    nba_data = extract_training_data("NBA")
    nba_award_model = train_award_prediction_model(nba_data['award_features'], "NBA")
    nba_team_model = train_team_success_model(nba_data['team_features'], "NBA")
    nba_player_model = train_player_performance_model(nba_data['player_features'], "NBA")
    nba_injury_model = train_injury_risk_model(nba_data['injury_features'], "NBA")
    save_model_metadata(nba_award_model, nba_team_model, nba_player_model, nba_injury_model, "NBA")
    
    # Train WNBA models
    wnba_data = extract_training_data("WNBA")
    wnba_award_model = train_award_prediction_model(wnba_data['award_features'], "WNBA")
    wnba_team_model = train_team_success_model(wnba_data['team_features'], "WNBA")
    wnba_player_model = train_player_performance_model(wnba_data['player_features'], "WNBA")
    wnba_injury_model = train_injury_risk_model(wnba_data['injury_features'], "WNBA")
    save_model_metadata(wnba_award_model, wnba_team_model, wnba_player_model, wnba_injury_model, "WNBA")

dag = model_training() 