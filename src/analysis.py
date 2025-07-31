"""
Statistical analysis module for weather and energy data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherEnergyAnalyzer:
    """Performs statistical analysis on weather and energy data."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.correlation_threshold = 0.7
        
    def calculate_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate correlations between temperature and energy consumption.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            DataFrame with correlation results
        """
        logger.info("Calculating temperature-energy correlations...")
        
        correlation_results = []
        
        for city_name, df in data.items():
            if len(df) < 10:  # Need minimum data for meaningful correlation
                logger.warning(f"Insufficient data for {city_name}: {len(df)} records")
                continue
                
            # Calculate various correlations
            correlations = {
                'city': city_name,
                'records': len(df),
                'max_temp_correlation': df['max_temp_f'].corr(df['energy_consumption_mwh']),
                'min_temp_correlation': df['min_temp_f'].corr(df['energy_consumption_mwh']),
                'avg_temp_correlation': df['avg_temp_f'].corr(df['energy_consumption_mwh']),
                'temp_range_correlation': df['temp_range'].corr(df['energy_consumption_mwh']) if 'temp_range' in df.columns else np.nan
            }
            
            # Calculate R-squared for average temperature
            if not df['avg_temp_f'].isna().all() and not df['energy_consumption_mwh'].isna().all():
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['avg_temp_f'].dropna(), 
                    df['energy_consumption_mwh'].dropna()
                )
                correlations.update({
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'std_error': std_err
                })
            
            # Determine if correlation is significant
            correlations['strong_correlation'] = abs(correlations['avg_temp_correlation']) > self.correlation_threshold
            correlations['correlation_strength'] = self._classify_correlation(correlations['avg_temp_correlation'])
            
            correlation_results.append(correlations)
        
        results_df = pd.DataFrame(correlation_results)
        logger.info(f"Correlation analysis completed for {len(results_df)} cities")
        
        return results_df
    
    def _classify_correlation(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return 'Very Strong'
        elif abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.5:
            return 'Moderate'
        elif abs_corr >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def analyze_seasonal_patterns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Analyze seasonal patterns in energy consumption.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            Dictionary of seasonal analysis results
        """
        logger.info("Analyzing seasonal patterns...")
        
        seasonal_results = {}
        
        for city_name, df in data.items():
            if len(df) < 30:  # Need minimum data for seasonal analysis
                continue
                
            # Add seasonal features
            df_seasonal = df.copy()
            df_seasonal['month'] = df_seasonal['date'].dt.month
            df_seasonal['season'] = df_seasonal['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Calculate seasonal statistics
            seasonal_stats = df_seasonal.groupby('season').agg({
                'avg_temp_f': ['mean', 'std', 'min', 'max'],
                'energy_consumption_mwh': ['mean', 'std', 'min', 'max'],
                'date': 'count'
            }).round(2)
            
            # Flatten column names
            seasonal_stats.columns = ['_'.join(col) for col in seasonal_stats.columns]
            seasonal_stats = seasonal_stats.reset_index()
            seasonal_stats['city'] = city_name
            
            seasonal_results[city_name] = seasonal_stats
        
        logger.info(f"Seasonal analysis completed for {len(seasonal_results)} cities")
        return seasonal_results
    
    def analyze_weekend_patterns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze weekend vs weekday energy consumption patterns.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            DataFrame with weekend analysis results
        """
        logger.info("Analyzing weekend vs weekday patterns...")
        
        weekend_results = []
        
        for city_name, df in data.items():
            if 'is_weekend' not in df.columns:
                logger.warning(f"Weekend data not available for {city_name}")
                continue
                
            # Calculate weekend vs weekday statistics
            weekend_stats = df.groupby('is_weekend').agg({
                'energy_consumption_mwh': ['mean', 'std', 'count'],
                'avg_temp_f': 'mean'
            }).round(2)
            
            # Calculate difference
            if len(weekend_stats) == 2:  # Both weekend and weekday data
                weekend_avg = weekend_stats.loc[True, ('energy_consumption_mwh', 'mean')]
                weekday_avg = weekend_stats.loc[False, ('energy_consumption_mwh', 'mean')]
                
                difference = weekend_avg - weekday_avg
                pct_difference = (difference / weekday_avg) * 100
                
                weekend_results.append({
                    'city': city_name,
                    'weekday_avg_mwh': weekday_avg,
                    'weekend_avg_mwh': weekend_avg,
                    'difference_mwh': difference,
                    'pct_difference': pct_difference,
                    'weekend_higher': difference > 0,
                    'weekday_count': weekend_stats.loc[False, ('energy_consumption_mwh', 'count')],
                    'weekend_count': weekend_stats.loc[True, ('energy_consumption_mwh', 'count')]
                })
        
        results_df = pd.DataFrame(weekend_results)
        logger.info(f"Weekend analysis completed for {len(results_df)} cities")
        
        return results_df
    
    def analyze_temperature_ranges(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Analyze energy consumption by temperature ranges.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            Dictionary of temperature range analysis results
        """
        logger.info("Analyzing energy consumption by temperature ranges...")
        
        temp_range_results = {}
        
        for city_name, df in data.items():
            if 'temp_category' not in df.columns:
                logger.warning(f"Temperature categories not available for {city_name}")
                continue
                
            # Calculate statistics by temperature range
            temp_stats = df.groupby('temp_category').agg({
                'energy_consumption_mwh': ['mean', 'std', 'count'],
                'avg_temp_f': ['mean', 'min', 'max']
            }).round(2)
            
            # Flatten column names
            temp_stats.columns = ['_'.join(col) for col in temp_stats.columns]
            temp_stats = temp_stats.reset_index()
            temp_stats['city'] = city_name
            
            temp_range_results[city_name] = temp_stats
        
        logger.info(f"Temperature range analysis completed for {len(temp_range_results)} cities")
        return temp_range_results
    
    def identify_outliers(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Identify outliers in energy consumption data.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            Dictionary of outlier analysis results
        """
        logger.info("Identifying outliers in energy consumption...")
        
        outlier_results = {}
        
        for city_name, df in data.items():
            # Use IQR method for outlier detection
            Q1 = df['energy_consumption_mwh'].quantile(0.25)
            Q3 = df['energy_consumption_mwh'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[
                (df['energy_consumption_mwh'] < lower_bound) | 
                (df['energy_consumption_mwh'] > upper_bound)
            ].copy()
            
            if len(outliers) > 0:
                outliers['outlier_type'] = np.where(
                    outliers['energy_consumption_mwh'] < lower_bound, 
                    'Low', 'High'
                )
                outliers['city'] = city_name
                outlier_results[city_name] = outliers
                
                logger.info(f"Found {len(outliers)} outliers for {city_name}")
        
        return outlier_results
    
    def generate_insights(self, correlation_results: pd.DataFrame, 
                         seasonal_results: Dict[str, pd.DataFrame],
                         weekend_results: pd.DataFrame) -> Dict[str, any]:
        """
        Generate business insights from analysis results.
        
        Args:
            correlation_results: Correlation analysis results
            seasonal_results: Seasonal analysis results
            weekend_results: Weekend analysis results
            
        Returns:
            Dictionary of insights
        """
        logger.info("Generating business insights...")
        
        insights = {
            'summary': {},
            'correlations': {},
            'seasonal': {},
            'weekend': {},
            'recommendations': []
        }
        
        # Correlation insights
        if len(correlation_results) > 0:
            strong_correlations = correlation_results[correlation_results['strong_correlation']]
            insights['correlations'] = {
                'cities_with_strong_correlation': len(strong_correlations),
                'average_correlation': correlation_results['avg_temp_correlation'].mean(),
                'strongest_correlation_city': correlation_results.loc[
                    correlation_results['avg_temp_correlation'].abs().idxmax(), 'city'
                ],
                'strongest_correlation_value': correlation_results['avg_temp_correlation'].abs().max()
            }
        
        # Weekend insights
        if len(weekend_results) > 0:
            insights['weekend'] = {
                'cities_weekend_higher': (weekend_results['weekend_higher']).sum(),
                'average_weekend_difference_pct': weekend_results['pct_difference'].mean(),
                'max_weekend_difference_city': weekend_results.loc[
                    weekend_results['pct_difference'].abs().idxmax(), 'city'
                ],
                'max_weekend_difference_pct': weekend_results['pct_difference'].abs().max()
            }
        
        # Generate recommendations
        if insights['correlations'].get('cities_with_strong_correlation', 0) > 0:
            insights['recommendations'].append(
                "Strong temperature-energy correlations detected. Consider temperature-based demand forecasting."
            )
        
        if insights['weekend'].get('average_weekend_difference_pct', 0) > 5:
            insights['recommendations'].append(
                "Significant weekend usage patterns detected. Implement day-of-week forecasting adjustments."
            )
        
        logger.info("Business insights generated")
        return insights
    
    def run_full_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Run comprehensive analysis on all data.
        
        Args:
            data: Dictionary of DataFrames by city
            
        Returns:
            Dictionary of all analysis results
        """
        logger.info("Running comprehensive analysis...")
        
        results = {
            'correlations': self.calculate_correlations(data),
            'seasonal': self.analyze_seasonal_patterns(data),
            'weekend': self.analyze_weekend_patterns(data),
            'temperature_ranges': self.analyze_temperature_ranges(data),
            'outliers': self.identify_outliers(data)
        }
        
        # Generate insights
        results['insights'] = self.generate_insights(
            results['correlations'], 
            results['seasonal'], 
            results['weekend']
        )
        
        logger.info("Comprehensive analysis completed")
        return results

def test_analyzer():
    """Test function for the analyzer."""
    print("Testing weather-energy analyzer...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    sample_data = {
        'Test City': pd.DataFrame({
            'date': dates,
            'avg_temp_f': np.random.normal(50, 20, 100),
            'max_temp_f': np.random.normal(60, 25, 100),
            'min_temp_f': np.random.normal(40, 15, 100),
            'energy_consumption_mwh': np.random.normal(15000, 2000, 100),
            'temp_range': np.random.normal(20, 5, 100),
            'is_weekend': np.random.choice([True, False], 100),
            'temp_category': np.random.choice(['<50°F', '50-60°F', '60-70°F', '70-80°F'], 100)
        })
    }
    
    # Test analyzer
    analyzer = WeatherEnergyAnalyzer()
    results = analyzer.run_full_analysis(sample_data)
    
    print(f"✓ Analysis test completed")
    print(f"  - Correlations: {len(results['correlations'])} cities")
    print(f"  - Seasonal patterns: {len(results['seasonal'])} cities")
    print(f"  - Weekend analysis: {len(results['weekend'])} cities")
    print(f"  - Insights generated: {len(results['insights']['recommendations'])} recommendations")