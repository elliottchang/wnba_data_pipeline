from airflow.decorators import dag, task
from pendulum import datetime, duration
import duckdb, pandas as pd, pathlib
import json
from typing import Dict, List
from datetime import timedelta

NBA_DB = pathlib.Path("/opt/airflow/data/nba_duck.db")
WNBA_DB = pathlib.Path("/opt/airflow/data/wnba_duck.db")

@dag(
    start_date=datetime(2025, 7, 1, tz="UTC"),
    schedule="0 11 * * *",      # run daily at 11:00 UTC (after quality checks)
    catchup=False,
    max_active_runs=1,
    tags=["dashboard", "analytics"]
)
def prepare_dashboard_data():
    """Prepare analytics data for dashboard consumption and visualization tools."""

    @task
    def extract_dashboard_data(league: str) -> Dict[str, pd.DataFrame]:
        """Extract all relevant data for dashboard preparation."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Extract analytics data
        player_agg = con.execute("SELECT * FROM analytics.player_aggregations").df()
        player_advanced = con.execute("SELECT * FROM analytics.player_advanced_metrics").df()
        player_rolling = con.execute("SELECT * FROM analytics.player_rolling_averages").df()
        
        # Extract ML features
        try:
            award_features = con.execute("SELECT * FROM ml_features.award_prediction_features").df()
            team_features = con.execute("SELECT * FROM ml_features.team_success_features").df()
            player_features = con.execute("SELECT * FROM ml_features.player_performance_features").df()
        except:
            award_features = pd.DataFrame()
            team_features = pd.DataFrame()
            player_features = pd.DataFrame()
        
        con.close()
        
        return {
            'player_aggregations': player_agg,
            'player_advanced': player_advanced,
            'player_rolling': player_rolling,
            'award_features': award_features,
            'team_features': team_features,
            'player_features': player_features
        }

    @task
    def create_player_dashboard_data(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create player-level dashboard data with key metrics and rankings."""
        player_agg = data_dict['player_aggregations']
        player_advanced = data_dict['player_advanced']
        award_features = data_dict['award_features']
        
        if player_agg.empty:
            return pd.DataFrame()
        
        # Merge all player data
        dashboard_data = player_agg.merge(player_advanced, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_NAME'], how='left')
        
        if not award_features.empty:
            dashboard_data = dashboard_data.merge(award_features[['PLAYER_ID', 'mvp_score', 'dpoy_score', 'sixmoy_score']], 
                                               on='PLAYER_ID', how='left')
        
        # Calculate rankings
        dashboard_data['ppg_rank'] = dashboard_data['PPG'].rank(ascending=False)
        dashboard_data['rpg_rank'] = dashboard_data['RPG'].rank(ascending=False)
        dashboard_data['apg_rank'] = dashboard_data['APG'].rank(ascending=False)
        dashboard_data['spg_rank'] = dashboard_data['SPG'].rank(ascending=False)
        dashboard_data['bpg_rank'] = dashboard_data['BPG'].rank(ascending=False)
        
        # Calculate efficiency metrics
        dashboard_data['efficiency_rating'] = (
            dashboard_data['FG_PCT'] * 0.4 +
            dashboard_data['FG3_PCT'] * 0.3 +
            dashboard_data['FT_PCT'] * 0.3
        )
        
        # Calculate overall player rating
        dashboard_data['overall_rating'] = (
            dashboard_data['PPG'] * 0.25 +
            dashboard_data['RPG'] * 0.15 +
            dashboard_data['APG'] * 0.15 +
            dashboard_data['SPG'] * 0.1 +
            dashboard_data['BPG'] * 0.1 +
            dashboard_data['efficiency_rating'] * 0.25
        )
        
        # Add league context
        dashboard_data['league'] = league
        dashboard_data['last_updated'] = pd.Timestamp.now()
        
        return dashboard_data

    @task
    def create_team_dashboard_data(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create team-level dashboard data with aggregated metrics."""
        player_agg = data_dict['player_aggregations']
        team_features = data_dict['team_features']
        
        if player_agg.empty:
            return pd.DataFrame()
        
        # Aggregate team-level metrics
        team_metrics = player_agg.groupby('TEAM_NAME').agg({
            'PPG': ['mean', 'sum'],
            'RPG': ['mean', 'sum'],
            'APG': ['mean', 'sum'],
            'SPG': ['mean', 'sum'],
            'BPG': ['mean', 'sum'],
            'FG_PCT': 'mean',
            'FG3_PCT': 'mean',
            'FT_PCT': 'mean',
            'PLAYER_ID': 'count'
        }).reset_index()
        
        # Flatten column names
        team_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in team_metrics.columns]
        
        # Calculate team offensive and defensive ratings
        team_metrics['offensive_rating'] = (
            team_metrics['PPG_mean'] * 0.4 +
            team_metrics['APG_mean'] * 0.3 +
            team_metrics['FG_PCT_mean'] * 100 * 0.3
        )
        
        team_metrics['defensive_rating'] = (
            team_metrics['SPG_mean'] * 0.4 +
            team_metrics['BPG_mean'] * 0.4 +
            team_metrics['RPG_mean'] * 0.2
        )
        
        # Calculate team depth score
        team_metrics['depth_score'] = team_metrics['PLAYER_ID_count'] * 10  # Simplified depth metric
        
        # Add league context
        team_metrics['league'] = league
        team_metrics['last_updated'] = pd.Timestamp.now()
        
        return team_metrics

    @task
    def create_league_dashboard_data(data_dict: Dict[str, pd.DataFrame], league: str) -> Dict[str, pd.DataFrame]:
        """Create league-wide dashboard data with trends and insights."""
        player_agg = data_dict['player_aggregations']
        player_rolling = data_dict['player_rolling']
        
        league_data = {}
        
        if not player_agg.empty:
            # League-wide statistics
            league_stats = {
                'total_players': len(player_agg),
                'avg_ppg': player_agg['PPG'].mean(),
                'avg_rpg': player_agg['RPG'].mean(),
                'avg_apg': player_agg['APG'].mean(),
                'avg_spg': player_agg['SPG'].mean(),
                'avg_bpg': player_agg['BPG'].mean(),
                'avg_fg_pct': player_agg['FG_PCT'].mean(),
                'avg_3p_pct': player_agg['FG3_PCT'].mean(),
                'avg_ft_pct': player_agg['FT_PCT'].mean(),
                'league': league,
                'last_updated': pd.Timestamp.now()
            }
            
            league_data['league_stats'] = pd.DataFrame([league_stats])
            
            # Top performers by category
            top_scorers = player_agg.nlargest(10, 'PPG')[['PLAYER_NAME', 'TEAM_NAME', 'PPG']]
            top_rebounders = player_agg.nlargest(10, 'RPG')[['PLAYER_NAME', 'TEAM_NAME', 'RPG']]
            top_assisters = player_agg.nlargest(10, 'APG')[['PLAYER_NAME', 'TEAM_NAME', 'APG']]
            top_stealers = player_agg.nlargest(10, 'SPG')[['PLAYER_NAME', 'TEAM_NAME', 'SPG']]
            top_blockers = player_agg.nlargest(10, 'BPG')[['PLAYER_NAME', 'TEAM_NAME', 'BPG']]
            
            league_data['top_scorers'] = top_scorers
            league_data['top_rebounders'] = top_rebounders
            league_data['top_assisters'] = top_assisters
            league_data['top_stealers'] = top_stealers
            league_data['top_blockers'] = top_blockers
            
            # Team rankings
            team_rankings = player_agg.groupby('TEAM_NAME').agg({
                'PPG': 'mean',
                'RPG': 'mean',
                'APG': 'mean',
                'SPG': 'mean',
                'BPG': 'mean'
            }).reset_index()
            
            team_rankings['offensive_rank'] = team_rankings['PPG'].rank(ascending=False)
            team_rankings['defensive_rank'] = (team_rankings['SPG'] + team_rankings['BPG']).rank(ascending=False)
            
            league_data['team_rankings'] = team_rankings
        
        return league_data

    @task
    def create_award_prediction_dashboard(data_dict: Dict[str, pd.DataFrame], league: str) -> pd.DataFrame:
        """Create dashboard data for award predictions."""
        award_features = data_dict['award_features']
        
        if award_features.empty:
            return pd.DataFrame()
        
        # Top MVP candidates
        mvp_candidates = award_features.nlargest(10, 'mvp_score')[['PLAYER_NAME', 'TEAM_NAME', 'mvp_score', 'PPG', 'RPG', 'APG']]
        
        # Top DPOY candidates
        dpoy_candidates = award_features.nlargest(10, 'dpoy_score')[['PLAYER_NAME', 'TEAM_NAME', 'dpoy_score', 'SPG', 'BPG', 'RPG']]
        
        # Top 6MOY candidates
        sixmoy_candidates = award_features.nlargest(10, 'sixmoy_score')[['PLAYER_NAME', 'TEAM_NAME', 'sixmoy_score', 'PPG', 'APG']]
        
        # Combine all award predictions
        award_predictions = pd.concat([
            mvp_candidates.assign(award='MVP'),
            dpoy_candidates.assign(award='DPOY'),
            sixmoy_candidates.assign(award='6MOY')
        ], ignore_index=True)
        
        award_predictions['league'] = league
        award_predictions['last_updated'] = pd.Timestamp.now()
        
        return award_predictions

    @task
    def create_kpi_summary(data_dict: Dict[str, pd.DataFrame], league: str) -> Dict[str, any]:
        """Create key performance indicators for executive dashboards."""
        player_agg = data_dict['player_aggregations']
        award_features = data_dict['award_features']
        
        kpis = {
            'league': league,
            'total_active_players': len(player_agg) if not player_agg.empty else 0,
            'avg_league_ppg': player_agg['PPG'].mean() if not player_agg.empty else 0,
            'top_scorer': player_agg.loc[player_agg['PPG'].idxmax(), 'PLAYER_NAME'] if not player_agg.empty else 'N/A',
            'top_scorer_ppg': player_agg['PPG'].max() if not player_agg.empty else 0,
            'mvp_favorite': award_features.loc[award_features['mvp_score'].idxmax(), 'PLAYER_NAME'] if not award_features.empty else 'N/A',
            'data_freshness': 'Current',
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return kpis

    @task
    def load_dashboard_data_to_duck(player_dashboard: pd.DataFrame, team_dashboard: pd.DataFrame,
                                   league_dashboard: Dict[str, pd.DataFrame], award_dashboard: pd.DataFrame,
                                   kpi_summary: Dict[str, any], league: str):
        """Load dashboard data to DuckDB for consumption by visualization tools."""
        db_path = NBA_DB if league == "NBA" else WNBA_DB
        con = duckdb.connect(db_path)
        
        # Create dashboard schema
        con.execute("CREATE SCHEMA IF NOT EXISTS dashboard;")
        
        # Load player dashboard data
        if not player_dashboard.empty:
            con.execute("DROP TABLE IF EXISTS dashboard.player_metrics")
            con.execute("CREATE TABLE dashboard.player_metrics AS SELECT * FROM player_dashboard")
        
        # Load team dashboard data
        if not team_dashboard.empty:
            con.execute("DROP TABLE IF EXISTS dashboard.team_metrics")
            con.execute("CREATE TABLE dashboard.team_metrics AS SELECT * FROM team_dashboard")
        
        # Load league dashboard data
        if league_dashboard:
            for table_name, df in league_dashboard.items():
                if not df.empty:
                    con.execute(f"DROP TABLE IF EXISTS dashboard.{table_name}")
                    con.execute(f"CREATE TABLE dashboard.{table_name} AS SELECT * FROM df")
        
        # Load award predictions
        if not award_dashboard.empty:
            con.execute("DROP TABLE IF EXISTS dashboard.award_predictions")
            con.execute("CREATE TABLE dashboard.award_predictions AS SELECT * FROM award_dashboard")
        
        # Store KPI summary as JSON
        kpi_json = json.dumps(kpi_summary)
        con.execute("DROP TABLE IF EXISTS dashboard.kpi_summary")
        con.execute(f"CREATE TABLE dashboard.kpi_summary AS SELECT '{kpi_json}' as kpi_data")
        
        con.close()

    @task
    def export_dashboard_data_to_json(player_dashboard: pd.DataFrame, team_dashboard: pd.DataFrame,
                                    league_dashboard: Dict[str, pd.DataFrame], award_dashboard: pd.DataFrame,
                                    kpi_summary: Dict[str, any], league: str):
        """Export dashboard data to JSON format for external dashboard tools."""
        
        # Create export directory
        export_dir = pathlib.Path("/opt/airflow/dashboard_exports")
        export_dir.mkdir(exist_ok=True)
        
        # Export player data
        if not player_dashboard.empty:
            player_dashboard.to_json(f"{export_dir}/{league.lower()}_player_dashboard.json", orient='records')
        
        # Export team data
        if not team_dashboard.empty:
            team_dashboard.to_json(f"{export_dir}/{league.lower()}_team_dashboard.json", orient='records')
        
        # Export league data
        for table_name, df in league_dashboard.items():
            if not df.empty:
                df.to_json(f"{export_dir}/{league.lower()}_{table_name}.json", orient='records')
        
        # Export award predictions
        if not award_dashboard.empty:
            award_dashboard.to_json(f"{export_dir}/{league.lower()}_award_predictions.json", orient='records')
        
        # Export KPI summary
        with open(f"{export_dir}/{league.lower()}_kpi_summary.json", 'w') as f:
            json.dump(kpi_summary, f, indent=2)

    # Process NBA data
    nba_data = extract_dashboard_data("NBA")
    nba_player_dashboard = create_player_dashboard_data(nba_data, "NBA")
    nba_team_dashboard = create_team_dashboard_data(nba_data, "NBA")
    nba_league_dashboard = create_league_dashboard_data(nba_data, "NBA")
    nba_award_dashboard = create_award_prediction_dashboard(nba_data, "NBA")
    nba_kpi_summary = create_kpi_summary(nba_data, "NBA")
    
    load_dashboard_data_to_duck(nba_player_dashboard, nba_team_dashboard, nba_league_dashboard, 
                               nba_award_dashboard, nba_kpi_summary, "NBA")
    export_dashboard_data_to_json(nba_player_dashboard, nba_team_dashboard, nba_league_dashboard,
                                 nba_award_dashboard, nba_kpi_summary, "NBA")
    
    # Process WNBA data
    wnba_data = extract_dashboard_data("WNBA")
    wnba_player_dashboard = create_player_dashboard_data(wnba_data, "WNBA")
    wnba_team_dashboard = create_team_dashboard_data(wnba_data, "WNBA")
    wnba_league_dashboard = create_league_dashboard_data(wnba_data, "WNBA")
    wnba_award_dashboard = create_award_prediction_dashboard(wnba_data, "WNBA")
    wnba_kpi_summary = create_kpi_summary(wnba_data, "WNBA")
    
    load_dashboard_data_to_duck(wnba_player_dashboard, wnba_team_dashboard, wnba_league_dashboard,
                               wnba_award_dashboard, wnba_kpi_summary, "WNBA")
    export_dashboard_data_to_json(wnba_player_dashboard, wnba_team_dashboard, wnba_league_dashboard,
                                 wnba_award_dashboard, wnba_kpi_summary, "WNBA")

dag = prepare_dashboard_data() 