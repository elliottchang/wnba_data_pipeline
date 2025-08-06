# WNBA/NBA Data Pipeline

A comprehensive data orchestration platform for WNBA and NBA analytics, designed to feed data into analytics dashboards and custom ML models.

## Architecture Overview

This pipeline ingests data from multiple sources and processes it through several layers:

### Data Sources
- **NBA API**: Box scores, play-by-play, player stats, team data
- **ESPN API**: Scoreboard data, game schedules
- **Basketball Reference**: Historical data, advanced stats
- **Custom APIs**: Additional basketball data sources

### Data Layers
1. **Raw Layer**: Unprocessed data from APIs
2. **Staging Layer**: Cleaned and validated data
3. **Analytics Layer**: Aggregated data for dashboards
4. **ML Layer**: Feature-engineered data for machine learning models

### Pipeline Components
- **Data Ingestion**: Daily/hourly data collection from APIs
- **Data Transformation**: Cleaning, validation, and aggregation
- **Data Quality**: Monitoring and alerting
- **ML Feature Engineering**: Creating features for predictive models
- **Dashboard Feeding**: Preparing data for visualization tools

## DAGs Overview

### Core Data Ingestion
- `ingest_nba_daily.py`: Daily NBA box scores and player stats
- `ingest_wnba_daily.py`: Daily WNBA data collection
- `ingest_play_by_play.py`: Detailed play-by-play data
- `ingest_team_data.py`: Team rosters, standings, schedules

### Data Processing
- `transform_player_stats.py`: Player performance aggregations
- `transform_team_stats.py`: Team-level analytics
- `transform_advanced_stats.py`: Advanced metrics calculation

### ML Pipeline
- `feature_engineering.py`: Create features for ML models
- `model_training.py`: Train predictive models
- `model_inference.py`: Generate predictions

### Data Quality
- `data_quality_checks.py`: Validate data integrity
- `data_freshness_monitoring.py`: Monitor data pipeline health

## Quick Start

1. **Start the Airflow environment**:
   ```bash
   astro dev start
   ```

2. **Access the Airflow UI**: http://localhost:8080

3. **Monitor DAGs**: Check the Airflow UI for pipeline status

## Data Models

### Player Analytics
- Individual player performance metrics
- Season-long aggregations
- Advanced stats (PER, VORP, etc.)
- Injury and availability tracking

### Team Analytics
- Team performance metrics
- Roster composition analysis
- Head-to-head statistics
- Season projections

### League Analytics
- League-wide trends
- Award predictions
- Playoff projections
- Historical comparisons

## ML Models

### Predictive Models
- **End-of-Season Awards**: MVP, DPOY, 6MOY predictions
- **Team Success**: Win-loss predictions, playoff chances
- **Player Performance**: Points, rebounds, assists projections
- **Injury Risk**: Player availability forecasting

### Model Features
- Historical performance data
- Team context and chemistry
- Advanced metrics and ratios
- External factors (schedule, rest days)

## Dashboard Integration

The pipeline prepares data for various dashboard types:
- **Player Dashboards**: Individual performance tracking
- **Team Dashboards**: Team analytics and projections
- **League Dashboards**: League-wide insights and trends
- **Executive Dashboards**: High-level KPIs and summaries

## Configuration

Update `airflow_settings.yaml` to configure:
- Database connections
- API credentials
- Alert configurations
- Custom variables

## Development

To add new data sources or transformations:
1. Create new DAG files in the `dags/` directory
2. Follow the established patterns for task definitions
3. Add appropriate tests in `tests/dags/`
4. Update documentation as needed