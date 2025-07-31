"""
Data fetching module for NOAA weather and EIA energy APIs.
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standard API response structure."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

class WeatherDataFetcher:
    """Fetches weather data from NOAA Climate Data Online API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('NOAA_API_KEY')
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2"
        self.session = requests.Session()
        self.session.headers.update({'token': self.api_key})
        
        if not self.api_key:
            raise ValueError("NOAA API key is required")
    
    def fetch_weather_data(self, station_id: str, start_date: str, end_date: str, 
                          max_retries: int = 3) -> APIResponse:
        """
        Fetch weather data for a specific station and date range.
        
        Args:
            station_id: NOAA station ID (e.g., 'GHCND:USW00094728')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            
        Returns:
            APIResponse object with weather data
        """
        url = f"{self.base_url}/data"
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': start_date,
            'enddate': end_date,
            'datatypeid': 'TMAX,TMIN',
            'limit': 1000,
            'units': 'standard'
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching weather data for {station_id} from {start_date} to {end_date}")
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and data['results']:
                        df = self._process_weather_data(data['results'])
                        logger.info(f"Successfully fetched {len(df)} weather records")
                        return APIResponse(success=True, data=df, status_code=200)
                    else:
                        logger.warning(f"No weather data found for {station_id}")
                        return APIResponse(success=False, error="No data found", status_code=200)
                        
                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return APIResponse(success=False, error=f"API error: {response.status_code}", 
                                     status_code=response.status_code)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return APIResponse(success=False, error=str(e))
                    
        return APIResponse(success=False, error="Max retries exceeded")
    
    def _process_weather_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Process raw weather data from NOAA API."""
        processed_records = []
        
        for record in raw_data:
            # Convert temperature from tenths of degrees Celsius to Fahrenheit
            temp_celsius = record['value'] / 10.0
            temp_fahrenheit = (temp_celsius * 9/5) + 32
            
            processed_records.append({
                'date': record['date'][:10],  # Extract date part
                'station_id': record['station'],
                'data_type': record['datatype'],
                'temperature_f': temp_fahrenheit,
                'temperature_c': temp_celsius
            })
        
        df = pd.DataFrame(processed_records)
        
        # Pivot to get TMAX and TMIN as separate columns
        df_pivot = df.pivot_table(
            index=['date', 'station_id'], 
            columns='data_type', 
            values='temperature_f',
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={'TMAX': 'max_temp_f', 'TMIN': 'min_temp_f'})
        
        # Add derived columns
        df_pivot['avg_temp_f'] = (df_pivot['max_temp_f'] + df_pivot['min_temp_f']) / 2
        df_pivot['date'] = pd.to_datetime(df_pivot['date'])
        
        return df_pivot

class EnergyDataFetcher:
    """Fetches energy data from EIA API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('EIA_API_KEY')
        self.base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data"
        self.session = requests.Session()
        
        if not self.api_key:
            raise ValueError("EIA API key is required")
    
    def fetch_energy_data(self, region_code: str, start_date: str, end_date: str,
                         max_retries: int = 3) -> APIResponse:
        """
        Fetch energy consumption data for a specific region and date range.
        
        Args:
            region_code: EIA region code (e.g., 'NYIS', 'PJM')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            
        Returns:
            APIResponse object with energy data
        """
        params = {
            'api_key': self.api_key,
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[respondent][]': region_code,
            'start': start_date,
            'end': end_date,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'desc',
            'offset': 0,
            'length': 5000
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching energy data for {region_code} from {start_date} to {end_date}")
                response = self.session.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'response' in data and 'data' in data['response']:
                        df = self._process_energy_data(data['response']['data'])
                        logger.info(f"Successfully fetched {len(df)} energy records")
                        return APIResponse(success=True, data=df, status_code=200)
                    else:
                        logger.warning(f"No energy data found for {region_code}")
                        return APIResponse(success=False, error="No data found", status_code=200)
                        
                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return APIResponse(success=False, error=f"API error: {response.status_code}",
                                     status_code=response.status_code)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return APIResponse(success=False, error=str(e))
                    
        return APIResponse(success=False, error="Max retries exceeded")
    
    def _process_energy_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Process raw energy data from EIA API."""
        processed_records = []
        
        for record in raw_data:
            processed_records.append({
                'date': record['period'],
                'region_code': record['respondent'],
                'energy_consumption_mwh': record['value'],
                'data_type': record.get('type-name', 'unknown')
            })
        
        df = pd.DataFrame(processed_records)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for demand data (actual consumption)
        df = df[df['data_type'].str.contains('Demand', case=False, na=False)]
        
        return df.sort_values('date')

def test_api_connections():
    """Test function to verify API connections."""
    print("Testing API connections...")
    
    # Test NOAA API
    try:
        weather_fetcher = WeatherDataFetcher()
        result = weather_fetcher.fetch_weather_data(
            station_id='GHCND:USW00094728',  # NYC
            start_date='2024-01-01',
            end_date='2024-01-05'
        )
        print(f"NOAA API test: {'✓ Success' if result.success else '✗ Failed'}")
        if not result.success:
            print(f"  Error: {result.error}")
    except Exception as e:
        print(f"NOAA API test: ✗ Failed - {str(e)}")
    
    # Test EIA API
    try:
        energy_fetcher = EnergyDataFetcher()
        result = energy_fetcher.fetch_energy_data(
            region_code='NYIS',
            start_date='2024-01-01',
            end_date='2024-01-05'
        )
        print(f"EIA API test: {'✓ Success' if result.success else '✗ Failed'}")
        if not result.success:
            print(f"  Error: {result.error}")
    except Exception as e:
        print(f"EIA API test: ✗ Failed - {str(e)}")

if __name__ == "__main__":
    test_api_connections()