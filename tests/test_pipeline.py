"""
Test suite for weather-energy analysis pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import WeatherDataFetcher, EnergyDataFetcher
from data_processor import DataProcessor
from analysis import WeatherEnergyAnalyzer
from pipeline import DataPipeline

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        self.sample_weather_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'temp_max_c': [20, 25, 22, 18, 30, 28, 24, 26, 23, 21],
            'temp_min_c': [10, 15, 12, 8, 20, 18, 14, 16, 13, 11],
            'city': ['TestCity'] * 10
        })
        
        self.sample_energy_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'energy_consumption_mwh': [15000, 16000, 14000, 18000, 19000, 17000, 15500, 16500, 14500, 17500],
            'city': ['TestCity'] * 10
        })
    
    def test_celsius_to_fahrenheit_conversion(self):
        """Test temperature conversion from Celsius to Fahrenheit."""
        processed_data = self.processor.clean_weather_data(self.sample_weather_data)
        
        # Check if Fahrenheit columns exist
        self.assertIn('max_temp_f', processed_data.columns)
        self.assertIn('min_temp_f', processed_data.columns)
        self.assertIn('avg_temp_f', processed_data.columns)
        
        # Check conversion accuracy (20°C should be 68°F)
        first_max_temp_f = processed_data['max_temp_f'].iloc[0]
        expected_temp_f = (20 * 9/5) + 32
        self.assertAlmostEqual(first_max_temp_f, expected_temp_f, places=1)
    
    def test_weekend_detection(self):
        """Test weekend detection functionality."""
        processed_data = self.processor.clean_weather_data(self.sample_weather_data)
        
        # Check if weekend column exists
        self.assertIn('is_weekend', processed_data.columns)
        
        # Check weekend detection logic
        weekend_mask = processed_data['date'].dt.dayofweek >= 5
        self.assertTrue((processed_data['is_weekend'] == weekend_mask).all())
    
    def test_data_merging(self):
        """Test weather and energy data merging."""
        weather_clean = self.processor.clean_weather_data(self.sample_weather_data)
        energy_clean = self.processor.clean_energy_data(self.sample_energy_data)
        
        merged_data = self.processor.merge_datasets(weather_clean, energy_clean)
        
        # Check if merge was successful
        self.assertIn('energy_consumption_mwh', merged_data.columns)
        self.assertIn('avg_temp_f', merged_data.columns)
        self.assertEqual(len(merged_data), 10)
    
    def test_outlier_detection(self):
        """Test outlier detection in energy data."""
        # Create data with outliers
        outlier_data = self.sample_energy_data.copy()
        outlier_data.loc[5, 'energy_consumption_mwh'] = 50000  # Clear outlier
        
        processed_data = self.processor.clean_energy_data(outlier_data)
        
        # Outlier should be flagged or handled
        self.assertLess(processed_data['energy_consumption_mwh'].max(), 50000)
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        missing_data = self.sample_weather_data.copy()
        missing_data.loc[3, 'temp_max_c'] = np.nan
        missing_data.loc[7, 'temp_min_c'] = np.nan
        
        processed_data = self.processor.clean_weather_data(missing_data)
        
        # Missing values should be handled (either filled or rows dropped)
        self.assertFalse(processed_data['max_temp_f'].isna().any())
        self.assertFalse(processed_data['min_temp_f'].isna().any())

class TestWeatherEnergyAnalyzer(unittest.TestCase):
    """Test cases for WeatherEnergyAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WeatherEnergyAnalyzer()
        
        # Create sample data with known correlation
        dates = pd.date_range('2024-01-01', periods=50)
        temperatures = np.random.normal(50, 20, 50)
        # Create energy consumption correlated with temperature
        energy_consumption = 15000 + (temperatures - 50) * 100 + np.random.normal(0, 500, 50)
        
        self.sample_data = {
            'TestCity': pd.DataFrame({
                'date': dates,
                'avg_temp_f': temperatures,
                'max_temp_f': temperatures + 5,
                'min_temp_f': temperatures - 5,
                'energy_consumption_mwh': energy_consumption,
                'is_weekend': np.random.choice([True, False], 50),
                'temp_category': np.random.choice(['<50°F', '50-60°F', '60-70°F', '>70°F'], 50)
            })
        }
    
    def test_correlation_calculation(self):
        """Test correlation calculation between temperature and energy."""
        correlation_results = self.analyzer.calculate_correlations(self.sample_data)
        
        # Check if results are returned
        self.assertGreater(len(correlation_results), 0)
        
        # Check if required columns exist
        required_cols = ['city', 'avg_temp_correlation', 'correlation_strength']
        for col in required_cols:
            self.assertIn(col, correlation_results.columns)
        
        # Check if correlation is reasonable (should be positive due to setup)
        correlation_value = correlation_results['avg_temp_correlation'].iloc[0]
        self.assertIsInstance(correlation_value, (int, float))
        self.assertGreaterEqual(correlation_value, -1)
        self.assertLessEqual(correlation_value, 1)
    
    def test_seasonal_analysis(self):
        """Test seasonal pattern analysis."""
        seasonal_results = self.analyzer.analyze_seasonal_patterns(self.sample_data)
        
        # Check if results are returned
        self.assertGreater(len(seasonal_results), 0)
        
        # Check if TestCity results exist
        self.assertIn('TestCity', seasonal_results)
        
        # Check if seasonal data has expected structure
        seasonal_df = seasonal_results['TestCity']
        self.assertIn('season', seasonal_df.columns)
        self.assertIn('city', seasonal_df.columns)
    
    def test_weekend_analysis(self):
        """Test weekend vs weekday analysis."""
        weekend_results = self.analyzer.analyze_weekend_patterns(self.sample_data)
        
        # Check if results are returned
        self.assertGreater(len(weekend_results), 0)
        
        # Check if required columns exist
        required_cols = ['city', 'weekday_avg_mwh', 'weekend_avg_mwh', 'pct_difference']
        for col in required_cols:
            self.assertIn(col, weekend_results.columns)
    
    def test_full_analysis(self):
        """Test comprehensive analysis pipeline."""
        results = self.analyzer.run_full_analysis(self.sample_data)
        
        # Check if all analysis components are present
        expected_keys = ['correlations', 'seasonal', 'weekend', 'insights']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check if insights are generated
        insights = results['insights']
        self.assertIn('correlations', insights)
        self.assertIn('recommendations', insights)

class TestDataPipeline(unittest.TestCase):
    """Test cases for DataPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'data_storage': {
                'raw_data_dir': os.path.join(self.temp_dir, 'raw'),
                'processed_data_dir': os.path.join(self.temp_dir, 'processed')
            },
            'cities': {
                'TestCity': {
                    'noaa_station_id': 'TEST123',
                    'eia_region_code': 'TEST',
                    'lat': 40.7128,
                    'lon': -74.0060
                }
            }
        }
        
        # Create directories
        os.makedirs(self.config['data_storage']['raw_data_dir'], exist_ok=True)
        os.makedirs(self.config['data_storage']['processed_data_dir'], exist_ok=True)
        
        self.pipeline = DataPipeline(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.processor)
        self.assertIsNotNone(self.pipeline.analyzer)
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Create sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'avg_temp_f': [70, 75, 80, 85, 90],
            'energy_consumption_mwh': [15000, 16000, 17000, 18000, 19000]
        })
        
        validation_result = self.pipeline.validate_data_quality(sample_data)
        self.assertTrue(validation_result['is_valid'])
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'avg_temp_f': [70, 200, 80, -100, 90],  # Invalid temperatures
            'energy_consumption_mwh': [15000, -1000, 17000, 18000, 19000]  # Negative energy
        })
        
        validation_result = self.pipeline.validate_data_quality(invalid_data)
        self.assertFalse(validation_result['is_valid'])

class TestDataFetchers(unittest.TestCase):
    """Test cases for data fetcher classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.weather_fetcher = WeatherDataFetcher()
        self.energy_fetcher = EnergyDataFetcher()
    
    def test_weather_fetcher_initialization(self):
        """Test weather fetcher initialization."""
        self.assertIsNotNone(self.weather_fetcher)
        self.assertIsNotNone(self.weather_fetcher.base_url)
    
    def test_energy_fetcher_initialization(self):
        """Test energy fetcher initialization."""
        self.assertIsNotNone(self.energy_fetcher)
        self.assertIsNotNone(self.energy_fetcher.base_url)
    
    def test_request_headers(self):
        """Test API request headers."""
        headers = self.weather_fetcher.get_headers()
        self.assertIn('token', headers)
        
        headers = self.energy_fetcher.get_headers()
        self.assertIn('X-Params', headers)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # This is a mock test - in real implementation, you'd test actual rate limiting
        self.assertTrue(hasattr(self.weather_fetcher, 'handle_rate_limiting'))
        self.assertTrue(hasattr(self.energy_fetcher, 'handle_rate_limiting'))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'data_storage': {
                'raw_data_dir': os.path.join(self.temp_dir, 'raw'),
                'processed_data_dir': os.path.join(self.temp_dir, 'processed')
            },
            'cities': {
                'TestCity': {
                    'noaa_station_id': 'TEST123',
                    'eia_region_code': 'TEST',
                    'lat': 40.7128,
                    'lon': -74.0060
                }
            },
            'data_quality': {
                'min_temp_f': -50,
                'max_temp_f': 130,
                'min_energy_mwh': 0,
                'max_missing_pct': 10
            }
        }
        
        # Create directories
        os.makedirs(self.config['data_storage']['raw_data_dir'], exist_ok=True)
        os.makedirs(self.config['data_storage']['processed_data_dir'], exist_ok=True)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_processing(self):
        """Test complete data processing pipeline."""
        # Create sample raw data files
        raw_weather_file = os.path.join(self.config['data_storage']['raw_data_dir'], 'testcity_weather.csv')
        raw_energy_file = os.path.join(self.config['data_storage']['raw_data_dir'], 'testcity_energy.csv')
        
        # Sample weather data
        weather_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'temp_max_c': np.random.normal(20, 5, 30),
            'temp_min_c': np.random.normal(10, 3, 30),
            'city': ['TestCity'] * 30
        })
        weather_data.to_csv(raw_weather_file, index=False)
        
        # Sample energy data
        energy_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'energy_consumption_mwh': np.random.normal(15000, 2000, 30),
            'city': ['TestCity'] * 30
        })
        energy_data.to_csv(raw_energy_file, index=False)
        
        # Initialize pipeline
        pipeline = DataPipeline(self.config)
        processor = DataProcessor()
        
        # Process data
        weather_clean = processor.clean_weather_data(weather_data)
        energy_clean = processor.clean_energy_data(energy_data)
        merged_data = processor.merge_datasets(weather_clean, energy_clean)
        
        # Check processing results
        self.assertGreater(len(merged_data), 0)
        self.assertIn('avg_temp_f', merged_data.columns)
        self.assertIn('energy_consumption_mwh', merged_data.columns)
        self.assertIn('is_weekend', merged_data.columns)
        
        # Test analysis
        analyzer = WeatherEnergyAnalyzer()
        analysis_results = analyzer.run_full_analysis({'TestCity': merged_data})
        
        # Check analysis results
        self.assertIn('correlations', analysis_results)
        self.assertIn('insights', analysis_results)
        
        # Test data quality validation
        validation_result = pipeline.validate_data_quality(merged_data)
        self.assertTrue(validation_result['is_valid'])

def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestWeatherEnergyAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestDataPipeline))
    test_suite.addTest(unittest.makeSuite(TestDataFetchers))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Weather-Energy Analysis Test Suite...")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)