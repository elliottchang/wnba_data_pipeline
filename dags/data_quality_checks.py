from airflow.decorators import dag, task
from pendulum import datetime, duration
import duckdb, pandas as pd, pathlib
import logging
from typing import Dict, List
from datetime import timedelta

NBA_DB = pathlib.Path("/opt/airflow/data/nba_duck.db")
WNBA_DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 10 * * *",      # run daily at 10:00 UTC (after feature engineering)
    catchup=False,
    max_active_runs=1,
    tags=["data-quality", "monitoring"]
)
def data_quality_checks():
    """Monitor data quality, freshness, and integrity across all data layers."""

    @task
    def check_data_freshness(league: str) -> dict:
        """Check if data is fresh and up-to-date."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        freshness_results = {
            'league': league,
            'raw_data_fresh': False,
            'analytics_data_fresh': False,
            'ml_features_fresh': False,
            'last_raw_update': None,
            'last_analytics_update': None,
            'last_ml_update': None
        }
        
        try:
            # Check raw data freshness
            if league == "NBA":
                raw_query = "SELECT MAX(created_at) as last_update FROM raw.nba_player_box_daily"
            else:
                raw_query = "SELECT MAX(created_at) as last_update FROM raw.wnba_player_box_daily"
            
            raw_result = con.execute(raw_query).df()
            if not raw_result.empty and raw_result['last_update'].iloc[0] is not None:
                last_raw = pd.to_datetime(raw_result['last_update'].iloc[0])
                freshness_results['last_raw_update'] = last_raw
                freshness_results['raw_data_fresh'] = (datetime.utcnow() - last_raw).days <= 1
            
            # Check analytics data freshness
            analytics_query = "SELECT MAX(last_updated) as last_update FROM analytics.player_aggregations"
            analytics_result = con.execute(analytics_query).df()
            if not analytics_result.empty and analytics_result['last_update'].iloc[0] is not None:
                last_analytics = pd.to_datetime(analytics_result['last_update'].iloc[0])
                freshness_results['last_analytics_update'] = last_analytics
                freshness_results['analytics_data_fresh'] = (datetime.utcnow() - last_analytics).days <= 1
            
            # Check ML features freshness
            ml_query = "SELECT MAX(feature_date) as last_update FROM ml_features.award_prediction_features"
            ml_result = con.execute(ml_query).df()
            if not ml_result.empty and ml_result['last_update'].iloc[0] is not None:
                last_ml = pd.to_datetime(ml_result['last_update'].iloc[0])
                freshness_results['last_ml_update'] = last_ml
                freshness_results['ml_features_fresh'] = (datetime.utcnow() - last_ml).days <= 1
                
        except Exception as e:
            logging.error(f"Error checking data freshness for {league}: {e}")
            freshness_results['error'] = str(e)
        
        con.close()
        return freshness_results

    @task
    def check_data_completeness(league: str) -> dict:
        """Check if all required data is present and complete."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        completeness_results = {
            'league': league,
            'raw_player_data_count': 0,
            'raw_team_data_count': 0,
            'analytics_player_count': 0,
            'analytics_advanced_count': 0,
            'ml_award_features_count': 0,
            'ml_team_features_count': 0,
            'ml_player_features_count': 0,
            'ml_injury_features_count': 0,
            'completeness_passed': True
        }
        
        try:
            # Check raw data completeness
            if league == "NBA":
                raw_player_query = "SELECT COUNT(*) as count FROM raw.nba_player_box_daily WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY"
                raw_team_query = "SELECT COUNT(*) as count FROM raw.nba_team_daily WHERE date >= CURRENT_DATE - INTERVAL 7 DAY"
            else:
                raw_player_query = "SELECT COUNT(*) as count FROM raw.wnba_player_box_daily WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY"
                raw_team_query = "SELECT COUNT(*) as count FROM raw.wnba_team_daily WHERE date >= CURRENT_DATE - INTERVAL 7 DAY"
            
            raw_player_result = con.execute(raw_player_query).df()
            raw_team_result = con.execute(raw_team_query).df()
            
            completeness_results['raw_player_data_count'] = raw_player_result['count'].iloc[0] if not raw_player_result.empty else 0
            completeness_results['raw_team_data_count'] = raw_team_result['count'].iloc[0] if not raw_team_result.empty else 0
            
            # Check analytics data completeness
            analytics_player_query = "SELECT COUNT(*) as count FROM analytics.player_aggregations"
            analytics_advanced_query = "SELECT COUNT(*) as count FROM analytics.player_advanced_metrics"
            
            analytics_player_result = con.execute(analytics_player_query).df()
            analytics_advanced_result = con.execute(analytics_advanced_query).df()
            
            completeness_results['analytics_player_count'] = analytics_player_result['count'].iloc[0] if not analytics_player_result.empty else 0
            completeness_results['analytics_advanced_count'] = analytics_advanced_result['count'].iloc[0] if not analytics_advanced_result.empty else 0
            
            # Check ML features completeness
            ml_award_query = "SELECT COUNT(*) as count FROM ml_features.award_prediction_features"
            ml_team_query = "SELECT COUNT(*) as count FROM ml_features.team_success_features"
            ml_player_query = "SELECT COUNT(*) as count FROM ml_features.player_performance_features"
            ml_injury_query = "SELECT COUNT(*) as count FROM ml_features.injury_risk_features"
            
            ml_award_result = con.execute(ml_award_query).df()
            ml_team_result = con.execute(ml_team_query).df()
            ml_player_result = con.execute(ml_player_query).df()
            ml_injury_result = con.execute(ml_injury_query).df()
            
            completeness_results['ml_award_features_count'] = ml_award_result['count'].iloc[0] if not ml_award_result.empty else 0
            completeness_results['ml_team_features_count'] = ml_team_result['count'].iloc[0] if not ml_team_result.empty else 0
            completeness_results['ml_player_features_count'] = ml_player_result['count'].iloc[0] if not ml_player_result.empty else 0
            completeness_results['ml_injury_features_count'] = ml_injury_result['count'].iloc[0] if not ml_injury_result.empty else 0
            
            # Determine if completeness check passed
            if completeness_results['raw_player_data_count'] == 0:
                completeness_results['completeness_passed'] = False
                completeness_results['error'] = 'No raw player data found'
            
        except Exception as e:
            logging.error(f"Error checking data completeness for {league}: {e}")
            completeness_results['completeness_passed'] = False
            completeness_results['error'] = str(e)
        
        con.close()
        return completeness_results

    @task
    def check_data_quality(league: str) -> dict:
        """Check data quality including null values, outliers, and data consistency."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        quality_results = {
            'league': league,
            'null_values_found': False,
            'outliers_detected': False,
            'data_consistency_passed': True,
            'quality_score': 1.0,
            'issues': []
        }
        
        try:
            # Check for null values in key columns
            if league == "NBA":
                null_check_query = """
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN PLAYER_NAME IS NULL THEN 1 ELSE 0 END) as null_player_names,
                    SUM(CASE WHEN POINTS IS NULL THEN 1 ELSE 0 END) as null_points,
                    SUM(CASE WHEN TEAM_NAME IS NULL THEN 1 ELSE 0 END) as null_team_names
                FROM raw.nba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY
                """
            else:
                null_check_query = """
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN PLAYER_NAME IS NULL THEN 1 ELSE 0 END) as null_player_names,
                    SUM(CASE WHEN POINTS IS NULL THEN 1 ELSE 0 END) as null_points,
                    SUM(CASE WHEN TEAM_NAME IS NULL THEN 1 ELSE 0 END) as null_team_names
                FROM raw.wnba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY
                """
            
            null_result = con.execute(null_check_query).df()
            
            if not null_result.empty:
                total_rows = null_result['total_rows'].iloc[0]
                null_player_names = null_result['null_player_names'].iloc[0]
                null_points = null_result['null_points'].iloc[0]
                null_team_names = null_result['null_team_names'].iloc[0]
                
                if null_player_names > 0 or null_points > 0 or null_team_names > 0:
                    quality_results['null_values_found'] = True
                    quality_results['issues'].append(f"Found {null_player_names} null player names, {null_points} null points, {null_team_names} null team names")
                
                # Calculate quality score
                total_issues = null_player_names + null_points + null_team_names
                quality_results['quality_score'] = max(0, 1 - (total_issues / (total_rows * 3)))
            
            # Check for statistical outliers (simplified)
            if league == "NBA":
                outlier_query = """
                SELECT 
                    AVG(POINTS) as avg_points,
                    STDDEV(POINTS) as std_points,
                    MAX(POINTS) as max_points,
                    MIN(POINTS) as min_points
                FROM raw.nba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY AND POINTS IS NOT NULL
                """
            else:
                outlier_query = """
                SELECT 
                    AVG(POINTS) as avg_points,
                    STDDEV(POINTS) as std_points,
                    MAX(POINTS) as max_points,
                    MIN(POINTS) as min_points
                FROM raw.wnba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY AND POINTS IS NOT NULL
                """
            
            outlier_result = con.execute(outlier_query).df()
            
            if not outlier_result.empty:
                avg_points = outlier_result['avg_points'].iloc[0]
                std_points = outlier_result['std_points'].iloc[0]
                max_points = outlier_result['max_points'].iloc[0]
                min_points = outlier_result['min_points'].iloc[0]
                
                # Check for extreme outliers (3+ standard deviations)
                upper_bound = avg_points + (3 * std_points)
                lower_bound = avg_points - (3 * std_points)
                
                if max_points > upper_bound or min_points < lower_bound:
                    quality_results['outliers_detected'] = True
                    quality_results['issues'].append(f"Detected outliers: max={max_points}, min={min_points}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
            
            # Check data consistency
            if league == "NBA":
                consistency_query = """
                SELECT 
                    COUNT(DISTINCT TEAM_NAME) as unique_teams,
                    COUNT(DISTINCT PLAYER_NAME) as unique_players
                FROM raw.nba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY
                """
            else:
                consistency_query = """
                SELECT 
                    COUNT(DISTINCT TEAM_NAME) as unique_teams,
                    COUNT(DISTINCT PLAYER_NAME) as unique_players
                FROM raw.wnba_player_box_daily 
                WHERE GAME_DATE >= CURRENT_DATE - INTERVAL 7 DAY
                """
            
            consistency_result = con.execute(consistency_query).df()
            
            if not consistency_result.empty:
                unique_teams = consistency_result['unique_teams'].iloc[0]
                unique_players = consistency_result['unique_players'].iloc[0]
                
                # Basic consistency checks
                expected_teams = 30 if league == "NBA" else 12
                if unique_teams < expected_teams * 0.5:  # At least 50% of teams should have data
                    quality_results['data_consistency_passed'] = False
                    quality_results['issues'].append(f"Only {unique_teams} teams found, expected ~{expected_teams}")
                
                if unique_players < 100:  # Should have significant number of players
                    quality_results['data_consistency_passed'] = False
                    quality_results['issues'].append(f"Only {unique_players} players found, expected more")
            
        except Exception as e:
            logging.error(f"Error checking data quality for {league}: {e}")
            quality_results['data_consistency_passed'] = False
            quality_results['issues'].append(f"Error: {str(e)}")
        
        con.close()
        return quality_results

    @task
    def generate_quality_report(nba_freshness: dict, wnba_freshness: dict,
                              nba_completeness: dict, wnba_completeness: dict,
                              nba_quality: dict, wnba_quality: dict) -> dict:
        """Generate a comprehensive data quality report."""
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'PASS',
            'nba_status': 'PASS',
            'wnba_status': 'PASS',
            'issues': [],
            'recommendations': []
        }
        
        # Check NBA status
        nba_issues = []
        if not nba_freshness.get('raw_data_fresh', False):
            nba_issues.append("NBA raw data not fresh")
        if not nba_completeness.get('completeness_passed', False):
            nba_issues.append("NBA data incomplete")
        if not nba_quality.get('data_consistency_passed', False):
            nba_issues.append("NBA data quality issues")
        
        if nba_issues:
            report['nba_status'] = 'FAIL'
            report['issues'].extend([f"NBA: {issue}" for issue in nba_issues])
        
        # Check WNBA status
        wnba_issues = []
        if not wnba_freshness.get('raw_data_fresh', False):
            wnba_issues.append("WNBA raw data not fresh")
        if not wnba_completeness.get('completeness_passed', False):
            wnba_issues.append("WNBA data incomplete")
        if not wnba_quality.get('data_consistency_passed', False):
            wnba_issues.append("WNBA data quality issues")
        
        if wnba_issues:
            report['wnba_status'] = 'FAIL'
            report['issues'].extend([f"WNBA: {issue}" for issue in wnba_issues])
        
        # Overall status
        if report['nba_status'] == 'FAIL' or report['wnba_status'] == 'FAIL':
            report['overall_status'] = 'FAIL'
        
        # Generate recommendations
        if report['overall_status'] == 'FAIL':
            report['recommendations'].append("Review data ingestion pipeline for failures")
            report['recommendations'].append("Check API connectivity and rate limits")
            report['recommendations'].append("Verify data transformation logic")
        
        # Log the report
        logging.info(f"Data Quality Report: {report}")
        
        return report

    # Run quality checks for both leagues
    nba_freshness = check_data_freshness("NBA")
    wnba_freshness = check_data_freshness("WNBA")
    
    nba_completeness = check_data_completeness("NBA")
    wnba_completeness = check_data_completeness("WNBA")
    
    nba_quality = check_data_quality("NBA")
    wnba_quality = check_data_quality("WNBA")
    
    # Generate comprehensive report
    quality_report = generate_quality_report(
        nba_freshness, wnba_freshness,
        nba_completeness, wnba_completeness,
        nba_quality, wnba_quality
    )

dag = data_quality_checks() 