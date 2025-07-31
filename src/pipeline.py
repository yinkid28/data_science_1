"""
Main data pipeline orchestration module.
"""

import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from pathlib import Path

from data_fetcher import WeatherDataFetcher, EnergyDataFetcher
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline for weather and energy data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cities = self.config['cities']
        self.weather_fetcher = WeatherDataFetcher()
        self.energy_fetcher = EnergyDataFetcher()
        self.processor = DataProcessor(config_path)
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.config['paths']['raw_data'],
            self.config['paths']['processed_data'],
            self.config['paths']['logs']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def fetch_historical_data(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all cities.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary of merged DataFrames by city
        """
        logger.info(f"Fetching {days} days of historical data for all cities...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        historical_data = {}
        
        for city in self.cities:
            city_name = city['name']
            logger.info(f"Processing historical data for {city_name}")
            
            try:
                # Fetch weather data
                weather_response = self.weather_fetcher.fetch_weather_data(
                    station_id=city['noaa_station_id'],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                # Fetch energy data
                energy_response = self.energy_fetcher.fetch_energy_data(
                    region_code=city['eia_region_code'],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                # Process and merge data if both successful
                if weather_response.success and energy_response.success:
                    # Clean data
                    clean_weather = self.processor.clean_weather_data(weather_response.data)
                    clean_energy = self.processor.clean_energy_data(energy_response.data)
                    
                    # Merge datasets
                    merged_data = self.processor.merge_datasets(
                        clean_weather, clean_energy, city_name
                    )
                    
                    historical_data[city_name] = merged_data
                    
                    # Save raw data
                    self._save_raw_data(weather_response.data, energy_response.data, city_name)
                    
                    logger.info(f"Successfully processed {len(merged_data)} records for {city_name}")
                    
                else:
                    logger.error(f"Failed to fetch data for {city_name}")
                    if not weather_response.success:
                        logger.error(f"  Weather API error: {weather_response.error}")
                    if not energy_response.success:
                        logger.error(f"  Energy API error: {energy_response.error}")
                        
            except Exception as e:
                logger.error(f"Error processing {city_name}: {str(e)}")
                continue
        
        logger.info(f"Historical data fetch completed for {len(historical_data)} cities")
        return historical_data
    
    def fetch_daily_data(self, target_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for all cities.
        
        Args:
            target_date: Specific date to fetch (defaults to yesterday)
            
        Returns:
            Dictionary of merged DataFrames by city
        """
        if target_date is None:
            target_date = datetime.now().date() - timedelta(days=1)
        
        logger.info(f"Fetching daily data for {target_date}")
        
        daily_data = {}
        
        for city in self.cities:
            city_name = city['name']
            logger.info(f"Processing daily data for {city_name}")
            
            try:
                # Fetch weather data
                weather_response = self.weather_fetcher.fetch_weather_data(
                    station_id=city['noaa_station_id'],
                    start_date=target_date.strftime('%Y-%m-%d'),
                    end_date=target_date.strftime('%Y-%m-%d')
                )
                
                # Fetch energy data
                energy_response = self.energy_fetcher.fetch_energy_data(
                    region_code=city['eia_region_code'],
                    start_date=target_date.strftime('%Y-%m-%d'),
                    end_date=target_date.strftime('%Y-%m-%d')
                )
                
                # Process and merge data if both successful
                if weather_response.success and energy_response.success:
                    # Clean data
                    clean_weather = self.processor.clean_weather_data(weather_response.data)
                    clean_energy = self.processor.clean_energy_data(energy_response.data)
                    
                    # Merge datasets
                    merged_data = self.processor.merge_datasets(
                        clean_weather, clean_energy, city_name
                    )
                    
                    daily_data[city_name] = merged_data
                    
                    # Update stored data
                    self._update_stored_data(merged_data, city_name)
                    
                    logger.info(f"Successfully processed daily data for {city_name}")
                    
                else:
                    logger.error(f"Failed to fetch daily data for {city_name}")
                    if not weather_response.success:
                        logger.error(f"  Weather API error: {weather_response.error}")
                    if not energy_response.success:
                        logger.error(f"  Energy API error: {energy_response.error}")
                        
            except Exception as e:
                logger.error(f"Error processing daily data for {city_name}: {str(e)}")
                continue
        
        logger.info(f"Daily data fetch completed for {len(daily_data)} cities")
        return daily_data
    
    def _save_raw_data(self, weather_data: pd.DataFrame, energy_data: pd.DataFrame, city_name: str):
        """Save raw data to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_data_path = Path(self.config['paths']['raw_data'])
        
        # Save weather data
        weather_file = raw_data_path / f"{city_name}_weather_{timestamp}.csv"
        weather_data.to_csv(weather_file, index=False)
        
        # Save energy data
        energy_file = raw_data_path / f"{city_name}_energy_{timestamp}.csv"
        energy_data.to_csv(energy_file, index=False)
        
        logger.info(f"Raw data saved for {city_name}")
    
    def _update_stored_data(self, merged_data: pd.DataFrame, city_name: str):
         """Update stored processed data."""
         processed_data_path = Path(self.config['paths']['processed_data'])
         filename = city_name.lower().replace(" ", "_") + "_processed.csv"
         processed_file = processed_data_path / filename
     
         if processed_file.exists():
             # Load existing data
             existing_data = pd.read_csv(processed_file)
             existing_data['date'] = pd.to_datetime(existing_data['date'])
     
             # Append new data (avoiding duplicates)
             combined_data = pd.concat([existing_data, merged_data], ignore_index=True)
             combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
             combined_data = combined_data.sort_values('date')
     
             # Save updated data
             combined_data.to_csv(processed_file, index=False)
         else:
             # Save new data
             merged_data.to_csv(processed_file, index=False)
     
         logger.info(f"Processed data updated for {city_name}")

    
    def load_processed_data(self, city_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from files.
        
        Args:
            city_name: Specific city to load (loads all if None)
            
        Returns:
            Dictionary of DataFrames by city
        """
        processed_data_path = Path(self.config['paths']['processed_data'])
        data = {}
    
        if city_name:
            cities_to_load = [city_name]
        else:
            cities_to_load = [city['name'] for city in self.cities]
    
        for city in cities_to_load:
            # Normalize file name
            filename = city.lower().replace(" ", "_") + "_processed.csv"
            processed_file = processed_data_path / filename
    
            if processed_file.exists():
                df = pd.read_csv(processed_file)
                df['date'] = pd.to_datetime(df['date'])
                data[city] = df
                logger.info(f"Loaded {len(df)} records for {city}")
            else:
                logger.warning(f"No processed data found for {city}")
    
        return data

    
    def run_data_quality_checks(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Run comprehensive data quality checks.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            Dictionary of quality reports
        """
        logger.info("Running data quality checks...")
        
        quality_reports = {}
        
        for city_name, df in data.items():
            report = self.processor.validate_data_quality(df)
            quality_reports[city_name] = report
            
            logger.info(f"Quality check for {city_name}: {report.quality_score:.1f}/100")
            
            if report.issues:
                logger.warning(f"Issues found in {city_name}: {', '.join(report.issues)}")
        
        # Generate summary
        reports_list = list(quality_reports.values())
        city_names = list(quality_reports.keys())
        summary = self.processor.generate_quality_summary(reports_list, city_names)
        
        # Save quality report
        quality_file = Path(self.config['paths']['processed_data']) / 'quality_report.csv'
        summary.to_csv(quality_file, index=False)
        
        logger.info("Data quality checks completed")
        return quality_reports
    
    def run_full_pipeline(self, mode: str = 'historical'):
        """
        Run the complete data pipeline.
        
        Args:
            mode: 'historical' for initial setup, 'daily' for regular updates
        """
        logger.info(f"Starting full pipeline in {mode} mode")
        
        try:
            if mode == 'historical':
                # Fetch historical data
                historical_days = self.config['analysis']['historical_days']
                data = self.fetch_historical_data(historical_days)
                
                # Save processed data
                for city_name, df in data.items():
                    self._update_stored_data(df, city_name)
                
            elif mode == 'daily':
                # Fetch daily data
                data = self.fetch_daily_data()
                
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Run quality checks
            quality_reports = self.run_data_quality_checks(data)
            
            # Log summary
            total_records = sum(len(df) for df in data.values())
            avg_quality = sum(report.quality_score for report in quality_reports.values()) / len(quality_reports)
            
            logger.info(f"Pipeline completed successfully:")
            logger.info(f"  - Cities processed: {len(data)}")
            logger.info(f"  - Total records: {total_records}")
            logger.info(f"  - Average quality score: {avg_quality:.1f}/100")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='US Weather + Energy Data Pipeline')
    parser.add_argument('--mode', choices=['historical', 'daily'], default='historical',
                       help='Pipeline mode: historical or daily')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = DataPipeline(args.config)
    pipeline.run_full_pipeline(args.mode)

if __name__ == "__main__":
    main()