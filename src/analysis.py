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
                'max_temp_correlation': df['max_temp_f'].corr(df['energy_consumption_mwh']) if 'max_temp_f' in df.columns else np.nan,
                'min_temp_correlation': df['min_temp_f'].corr(df['energy_consumption_mwh']) if 'min_temp_f' in df.columns else np.nan,
                'avg_temp_correlation': df['avg_temp_f'].corr(df['energy_consumption_mwh']) if 'avg_temp_f' in df.columns else np.nan,
                'temp_range_correlation': df['temp_range'].corr(df['energy_consumption_mwh']) if 'temp_range' in df.columns else np.nan
            }
            
            # Calculate R-squared and p-value for average temperature
            if 'avg_temp_f' in df.columns and 'energy_consumption_mwh' in df.columns:
                # Clean the data first - remove rows where either value is NaN
                clean_data = df[['avg_temp_f', 'energy_consumption_mwh']].dropna()
                
                if len(clean_data) >= 3:  # Need at least 3 points for meaningful stats
                    temp_values = clean_data['avg_temp_f'].values
                    energy_values = clean_data['energy_consumption_mwh'].values
                    
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(temp_values, energy_values)
                        
                        # Format p-value for display
                        if p_value < 1e-10:
                            p_value_display = "< 0.001"
                        elif p_value < 0.001:
                            p_value_display = f"{p_value:.2e}"
                        elif p_value < 0.01:
                            p_value_display = f"{p_value:.4f}"
                        else:
                            p_value_display = f"{p_value:.3f}"
                        
                        correlations.update({
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'p_value_display': p_value_display,
                            'slope': slope,
                            'intercept': intercept,
                            'std_error': std_err
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating regression for {city_name}: {e}")
                        correlations.update({
                            'r_squared': np.nan,
                            'p_value': np.nan,
                            'p_value_display': "Error",
                            'slope': np.nan,
                            'intercept': np.nan,
                            'std_error': np.nan
                        })
                else:
                    correlations.update({
                        'r_squared': np.nan,
                        'p_value': np.nan,
                        'p_value_display': "Insufficient data",
                        'slope': np.nan,
                        'intercept': np.nan,
                        'std_error': np.nan
                    })
            else:
                correlations.update({
                    'r_squared': np.nan,
                    'p_value': np.nan,
                    'p_value_display': "Missing columns",
                    'slope': np.nan,
                    'intercept': np.nan,
                    'std_error': np.nan
                })
            
            # Determine if correlation is significant
            avg_corr = correlations.get('avg_temp_correlation', 0)
            correlations['strong_correlation'] = abs(avg_corr) > self.correlation_threshold if not pd.isna(avg_corr) else False
            correlations['correlation_strength'] = self._classify_correlation(avg_corr)
            
            correlation_results.append(correlations)
        
        results_df = pd.DataFrame(correlation_results)
        logger.info(f"Correlation analysis completed for {len(results_df)} cities")
        
        return results_df
    
    def _classify_correlation(self, correlation: float) -> str:
        """Classify correlation strength."""
        if pd.isna(correlation):
            return 'No Data'
        
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
            # Filter out rows with NaN correlations
            valid_correlations = correlation_results.dropna(subset=['avg_temp_correlation'])
            
            if len(valid_correlations) > 0:
                strong_correlations = valid_correlations[valid_correlations['strong_correlation']]
                
                # Find strongest correlation (by absolute value)
                strongest_idx = valid_correlations['avg_temp_correlation'].abs().idxmax()
                strongest_city = valid_correlations.loc[strongest_idx, 'city']
                strongest_value = valid_correlations.loc[strongest_idx, 'avg_temp_correlation']
                
                insights['correlations'] = {
                    'cities_with_strong_correlation': len(strong_correlations),
                    'average_correlation': valid_correlations['avg_temp_correlation'].mean(),
                    'strongest_correlation_city': strongest_city,
                    'strongest_correlation_value': abs(strongest_value)
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