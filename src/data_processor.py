"""
Data processing and quality validation module.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Data quality assessment results."""
    total_records: int
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    data_freshness: Dict[str, int]
    quality_score: float
    issues: List[str]

class DataProcessor:
    """Processes and validates weather and energy data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.quality_thresholds = self.config['data_quality']
        
    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate weather data.
        
        Args:
            df: Raw weather DataFrame
            
        Returns:
            Cleaned weather DataFrame
        """
        logger.info("Cleaning weather data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Ensure date column is datetime
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Handle missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['max_temp_f', 'min_temp_f'])
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to missing temperature data")
        
        # Validate temperature ranges
        temp_min = self.quality_thresholds['temperature']['min_fahrenheit']
        temp_max = self.quality_thresholds['temperature']['max_fahrenheit']
        
        # Flag outliers
        temp_outliers = (
            (df_clean['max_temp_f'] < temp_min) | 
            (df_clean['max_temp_f'] > temp_max) |
            (df_clean['min_temp_f'] < temp_min) | 
            (df_clean['min_temp_f'] > temp_max)
        )
        
        if temp_outliers.any():
            logger.warning(f"Found {temp_outliers.sum()} temperature outliers")
            df_clean.loc[temp_outliers, 'quality_flag'] = 'temperature_outlier'
        
        # Ensure min_temp <= max_temp
        temp_logic_error = df_clean['min_temp_f'] > df_clean['max_temp_f']
        if temp_logic_error.any():
            logger.warning(f"Found {temp_logic_error.sum()} records where min > max temperature")
            df_clean.loc[temp_logic_error, 'quality_flag'] = 'temperature_logic_error'
        
        # Recalculate average temperature
        df_clean['avg_temp_f'] = (df_clean['max_temp_f'] + df_clean['min_temp_f']) / 2
        
        # Add day of week and weekend indicator
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])
        df_clean['day_name'] = df_clean['date'].dt.day_name()
        
        # Add temperature category for analysis
        df_clean['temp_category'] = pd.cut(
            df_clean['avg_temp_f'],
            bins=[-float('inf'), 50, 60, 70, 80, 90, float('inf')],
            labels=['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        )
        
        logger.info(f"Weather data cleaned: {len(df_clean)} records remaining")
        return df_clean
    
    def clean_energy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate energy consumption data.
        
        Args:
            df: Raw energy DataFrame
            
        Returns:
            Cleaned energy DataFrame
        """
        logger.info("Cleaning energy data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Ensure date column is datetime
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Convert to numeric (handles bad strings)
        df_clean['energy_consumption_mwh'] = pd.to_numeric(df_clean['energy_consumption_mwh'], errors='coerce')
        
        # Handle missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['energy_consumption_mwh'])
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to missing or invalid energy data")
        
        # Validate energy consumption (should be positive)
        min_consumption = self.quality_thresholds['energy']['min_consumption']
        negative_consumption = df_clean['energy_consumption_mwh'] < min_consumption
        
        if negative_consumption.any():
            logger.warning(f"Found {negative_consumption.sum()} records with negative energy consumption")
            df_clean.loc[negative_consumption, 'quality_flag'] = 'negative_consumption'
        
        # Detect extreme outliers using IQR method
        Q1 = df_clean['energy_consumption_mwh'].quantile(0.25)
        Q3 = df_clean['energy_consumption_mwh'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (
            (df_clean['energy_consumption_mwh'] < lower_bound) | 
            (df_clean['energy_consumption_mwh'] > upper_bound)
        )
        
        if outliers.any():
            logger.warning(f"Found {outliers.sum()} energy consumption outliers")
            df_clean.loc[outliers, 'quality_flag'] = 'energy_outlier'
        
        # Add day of week and weekend indicator
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])
        df_clean['day_name'] = df_clean['date'].dt.day_name()
        
        # Add rolling averages for trend analysis
        df_clean = df_clean.sort_values('date')
        df_clean['energy_7day_avg'] = df_clean['energy_consumption_mwh'].rolling(window=7, center=True).mean()
        df_clean['energy_30day_avg'] = df_clean['energy_consumption_mwh'].rolling(window=30, center=True).mean()
        
        logger.info(f"Energy data cleaned: {len(df_clean)} records remaining")
        return df_clean

    
    def merge_datasets(self, weather_df: pd.DataFrame, energy_df: pd.DataFrame,
                      city_name: str) -> pd.DataFrame:
        """
        Merge weather and energy datasets on date.
        
        Args:
            weather_df: Cleaned weather DataFrame
            energy_df: Cleaned energy DataFrame
            city_name: Name of the city for labeling
            
        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging datasets for {city_name}")
        
        # Prepare weather data for merge
        weather_merge = weather_df.copy()
        weather_merge['date'] = weather_merge['date'].dt.date
        
        # Prepare energy data for merge
        energy_merge = energy_df.copy()
        energy_merge['date'] = energy_merge['date'].dt.date
        
        # Merge on date
        merged_df = pd.merge(
            weather_merge,
            energy_merge,
            on='date',
            how='inner',
            suffixes=('_weather', '_energy')
        )
        
        # Add city information
        merged_df['city'] = city_name
        
        # Calculate daily statistics
        merged_df['temp_range'] = merged_df['max_temp_f'] - merged_df['min_temp_f']
        
        # Handle weekend flags (take from weather data if both exist)
        if 'is_weekend_weather' in merged_df.columns and 'is_weekend_energy' in merged_df.columns:
            merged_df['is_weekend'] = merged_df['is_weekend_weather']
            merged_df['day_name'] = merged_df['day_name_weather']
            merged_df = merged_df.drop(['is_weekend_weather', 'is_weekend_energy', 
                                     'day_name_weather', 'day_name_energy'], axis=1)
        
        # Convert date back to datetime for consistency
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        logger.info(f"Merged dataset created: {len(merged_df)} records for {city_name}")
        return merged_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataQualityReport with quality metrics
        """
        logger.info("Validating data quality...")
        
        total_records = len(df)
        issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}
        
        # Check data freshness
        if 'date' in df.columns:
            latest_date = df['date'].max()
            days_old = (datetime.now().date() - latest_date.date()).days
            max_days_old = self.quality_thresholds['freshness']['max_days_old']
            
            data_freshness = {'days_old': days_old, 'is_fresh': days_old <= max_days_old}
            
            if days_old > max_days_old:
                issues.append(f"Data is {days_old} days old (threshold: {max_days_old} days)")
        else:
            data_freshness = {'days_old': None, 'is_fresh': True}
        
        # Check for outliers
        outliers = {}
        if 'quality_flag' in df.columns:
            outlier_counts = df['quality_flag'].value_counts().to_dict()
            outliers.update(outlier_counts)
            
            if outlier_counts:
                issues.append(f"Found quality flags: {outlier_counts}")
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        
        # Penalize for missing values
        if missing_values:
            missing_rate = sum(missing_values.values()) / (total_records * len(df.columns))
            quality_score -= missing_rate * 50
        
        # Penalize for outliers
        if outliers:
            outlier_rate = sum(outliers.values()) / total_records
            quality_score -= outlier_rate * 30
        
        # Penalize for stale data
        if not data_freshness['is_fresh']:
            quality_score -= 20
        
        quality_score = max(0, quality_score)
        
        report = DataQualityReport(
            total_records=total_records,
            missing_values=missing_values,
            outliers=outliers,
            data_freshness=data_freshness,
            quality_score=quality_score,
            issues=issues
        )
        
        logger.info(f"Data quality score: {quality_score:.1f}/100")
        return report
    
    def generate_quality_summary(self, reports: List[DataQualityReport], 
                               city_names: List[str]) -> pd.DataFrame:
        """
        Generate summary of data quality across all cities.
        
        Args:
            reports: List of DataQualityReport objects
            city_names: List of city names corresponding to reports
            
        Returns:
            DataFrame with quality summary
        """
        summary_data = []
        
        for city, report in zip(city_names, reports):
            summary_data.append({
                'city': city,
                'total_records': report.total_records,
                'quality_score': report.quality_score,
                'missing_values': len(report.missing_values),
                'outliers': len(report.outliers),
                'days_old': report.data_freshness.get('days_old', 0),
                'is_fresh': report.data_freshness.get('is_fresh', True),
                'issues_count': len(report.issues)
            })
        
        return pd.DataFrame(summary_data)

def test_data_processor():
    """Test function for data processor."""
    print("Testing data processor...")
    
    # Create sample data
    sample_weather = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'station_id': ['GHCND:USW00094728'] * 10,
        'max_temp_f': [45, 48, 52, 55, 58, 60, 62, 65, 68, 70],
        'min_temp_f': [35, 38, 42, 45, 48, 50, 52, 55, 58, 60]
    })
    
    sample_energy = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'region_code': ['NYIS'] * 10,
        'energy_consumption_mwh': [15000, 16000, 14500, 15500, 16500, 15800, 14200, 15200, 16200, 15600],
        'data_type': ['Demand'] * 10
    })
    
    # Test processor
    processor = DataProcessor()
    
    # Clean data
    clean_weather = processor.clean_weather_data(sample_weather)
    clean_energy = processor.clean_energy_data(sample_energy)
    
    # Merge data
    merged_data = processor.merge_datasets(clean_weather, clean_energy, "New York")
    
    # Validate quality
    quality_report = processor.validate_data_quality(merged_data)
    
    print(f"✓ Data processing test completed")
    print(f"  - Weather records: {len(clean_weather)}")
    print(f"  - Energy records: {len(clean_energy)}")
    print(f"  - Merged records: {len(merged_data)}")
    print(f"  - Quality score: {quality_report.quality_score:.1f}/100")

if __name__ == "__main__":
    test_data_processor()