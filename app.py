"""
Streamlit dashboard for weather and energy data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import yaml
import os
import sys
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipeline import DataPipeline
    from analysis import WeatherEnergyAnalyzer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Weather & Energy Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .correlation-strong {
        color: #28a745;
        font-weight: bold;
    }
    .correlation-weak {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class WeatherEnergyDashboard:
    """Main dashboard class for weather and energy analysis."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.pipeline = None
        self.analyzer = None
        self.data = {}
        self.config = self._load_config()
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        """Load configuration file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Handle different config structures
            if isinstance(config, list):
                # If config is a list, try to get the first dictionary
                if len(config) > 0 and isinstance(config[0], dict):
                    return config[0]
                else:
                    st.warning("Config file structure is unexpected. Using default configuration.")
                    return self._get_default_config()
            elif isinstance(config, dict):
                return config
            else:
                st.warning("Config file structure is unexpected. Using default configuration.")
                return self._get_default_config()
                
        except FileNotFoundError:
            st.error("Configuration file not found. Please ensure config/config.yaml exists.")
            return self._get_default_config()
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'cities': {
                'New York': {'lat': 40.7128, 'lon': -74.0060, 'timezone': 'America/New_York'},
                'Los Angeles': {'lat': 34.0522, 'lon': -118.2437, 'timezone': 'America/Los_Angeles'},
                'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'timezone': 'America/Chicago'},
                'Houston': {'lat': 29.7604, 'lon': -95.3698, 'timezone': 'America/Chicago'},
                'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'timezone': 'America/Phoenix'},
                'Philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'timezone': 'America/New_York'},
                'San Antonio': {'lat': 29.4241, 'lon': -98.4936, 'timezone': 'America/Chicago'},
                'San Diego': {'lat': 32.7157, 'lon': -117.1611, 'timezone': 'America/Los_Angeles'},
                'Dallas': {'lat': 32.7767, 'lon': -96.7970, 'timezone': 'America/Chicago'},
                'San Jose': {'lat': 37.3382, 'lon': -121.8863, 'timezone': 'America/Los_Angeles'}
            },
            'api': {
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'timeout': 30,
                'retry_attempts': 3
            },
            'analysis': {
                'forecast_days': 7,
                'confidence_interval': 0.95,
                'seasonal_periods': [24, 168, 8760]
            }
        }
    
    def _initialize_components(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
            self.pipeline = DataPipeline(config_path)
            self.analyzer = WeatherEnergyAnalyzer()
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            raise e  # Add this line to reveal the error

    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_data(_self, start_date: str, end_date: str, cities: List[str]) -> Dict[str, pd.DataFrame]:
        """Load and cache data for selected date range and cities."""
        try:
            # Load processed data
            data = {}
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
            
            for city in cities:
                city_file = os.path.join(data_dir, f"{city.lower().replace(' ', '_')}_processed.csv")
                if os.path.exists(city_file):
                    df = pd.read_csv(city_file)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Filter by date range
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    filtered_df = df[mask].copy()
                    
                    if len(filtered_df) > 0:
                        data[city] = filtered_df
                    else:
                        st.warning(f"No data found for {city} in selected date range")
                else:
                    st.warning(f"No processed data file found for {city}")
            
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return {}
    
    def render_sidebar(self) -> Dict:
        """Render sidebar controls and return selected parameters."""
        st.sidebar.markdown('<div class="sidebar-header">Dashboard Controls</div>', unsafe_allow_html=True)
        
        # Date range selection
        st.sidebar.subheader("📅 Date Range")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        
        date_range = st.sidebar.date_input(
            "Select date range:",
            value=(start_date, end_date),
            min_value=datetime(2024, 1, 1).date(),
            max_value=end_date
        )
        
        # City selection - FIXED HERE
        st.sidebar.subheader("🏙️ Cities")
        
        # Handle cities as list of dictionaries (your actual config structure)
        if isinstance(self.config, dict) and 'cities' in self.config:
            if isinstance(self.config['cities'], list):
                # Extract city names from list of dictionaries
                available_cities = [city.get('name', f'City_{i}') for i, city in enumerate(self.config['cities'])]
            elif isinstance(self.config['cities'], dict):
                # Handle as dictionary (fallback)
                available_cities = list(self.config['cities'].keys())
            else:
                available_cities = []
        else:
            available_cities = []
        
        # Fallback to default cities if none found
        if not available_cities:
            st.sidebar.warning("No cities configured, using default cities")
            available_cities = ["New York", "Chicago", "Houston", "Phoenix", "Seattle"]
        
        selected_cities = st.sidebar.multiselect(
            "Select cities:",
            available_cities,
            default=available_cities[:3]  # Default to first 3 cities
        )
        
        # Analysis options
        st.sidebar.subheader("📊 Analysis Options")
        show_correlations = st.sidebar.checkbox("Show Correlation Analysis", value=True)
        show_seasonal = st.sidebar.checkbox("Show Seasonal Patterns", value=True)
        show_weekend = st.sidebar.checkbox("Show Weekend Analysis", value=True)
        show_outliers = st.sidebar.checkbox("Show Outlier Detection", value=False)
        
        # Refresh data button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        return {
            'date_range': date_range,
            'selected_cities': selected_cities,
            'show_correlations': show_correlations,
            'show_seasonal': show_seasonal,
            'show_weekend': show_weekend,
            'show_outliers': show_outliers
        }
    
    def render_header(self):
        """Render main header."""
        st.markdown('<h1 class="main-header">⚡ Weather & Energy Analysis Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_overview_metrics(self, data: Dict[str, pd.DataFrame]):
        """Render overview metrics."""
        st.subheader("📈 Overview Metrics")
        
        if not data:
            st.warning("No data available for selected cities and date range")
            return
        
        # Calculate summary metrics
        total_records = sum(len(df) for df in data.values())
        avg_temp = np.mean([df['avg_temp_f'].mean() for df in data.values() if 'avg_temp_f' in df.columns])
        avg_energy = np.mean([df['energy_consumption_mwh'].mean() for df in data.values() if 'energy_consumption_mwh' in df.columns])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Records",
                value=f"{total_records:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Cities Analyzed",
                value=len(data),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Avg Temperature",
                value=f"{avg_temp:.1f}°F" if not np.isnan(avg_temp) else "N/A",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Energy Usage",
                value=f"{avg_energy:,.0f} MWh" if not np.isnan(avg_energy) else "N/A",
                delta=None
            )
    
    def render_geographic_overview(self, data: Dict[str, pd.DataFrame]):
        """Render geographic overview visualization."""
        st.subheader("🗺️ Geographic Overview")
    
        if not data:
            st.warning("No data available for geographic overview")
            return
    
        # Create city summary data
        city_summary = []
        for city, df in data.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                previous = df.iloc[-2]  # compare with previous day
    
                # Get city info from config (list structure)
                city_info = {}
                if isinstance(self.config, dict) and 'cities' in self.config:
                    for city_config in self.config['cities']:
                        if city_config.get('name') == city:
                            # For now, use default lat/lon since your config doesn't have them
                            # You might need to add these to your config.yaml
                            city_info = {
                                'lat': 40.7128,  # Default to NYC coords
                                'lon': -74.0060
                            }
                            break
                else:
                    city_info = {}
    
                energy_today = latest.get('energy_consumption_mwh', 0)
                energy_yesterday = previous.get('energy_consumption_mwh', 0)
    
                # Calculate % change safely
                if energy_yesterday > 0:
                    percent_change = ((energy_today - energy_yesterday) / energy_yesterday) * 100
                else:
                    percent_change = 0.0
    
                city_summary.append({
                    'city': city,
                    'lat': city_info.get('lat', 40.7128),
                    'lon': city_info.get('lon', -74.0060),
                    'temperature': latest.get('avg_temp_f', 0),
                    'energy_usage': energy_today,
                    'percent_change': percent_change,
                    'date': latest.get('date', datetime.now())
                })
    
        if city_summary:
            summary_df = pd.DataFrame(city_summary)
    
            # Rank cities by energy usage
            summary_df['rank'] = summary_df['energy_usage'].rank(ascending=False)
    
            # Color code: red = high usage, green = low usage
            summary_df['color'] = summary_df['rank'].apply(
                lambda r: 'red' if r <= 2 else ('green' if r >= 4 else 'orange')
            )
    
            # Create map visualization
            fig = px.scatter_mapbox(
               summary_df,
               lat='lat',
               lon='lon',
               hover_name='city',
               hover_data={
                   'temperature': ':.1f',
                   'energy_usage': ':,.0f',
                   'percent_change': ':.1f',
                   'lat': False,
                   'lon': False
               },
               color='color',
               color_discrete_map={
                   'red': 'red',
                   'green': 'green',
                   'orange': 'orange'
               },
               size='energy_usage',
               size_max=30,
               zoom=3,
               height=400
           )

    
            fig.update_layout(
                mapbox_style="open-street-map",
                title="Current Energy Usage by City",
                showlegend=False  # not needed for discrete color
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
            # Show timestamp
            latest_timestamp = summary_df['date'].max()
            st.caption(f"Data last updated: {latest_timestamp.strftime('%B %d, %Y')}")




    
    def render_time_series_analysis(self, data: Dict[str, pd.DataFrame]):
        """Render time series analysis visualization."""
        st.subheader("📊 Time Series Analysis")
        
        if not data:
            st.warning("No data available for time series analysis")
            return
        
        # City selector
        selected_city = st.selectbox("Select city for detailed analysis:", list(data.keys()))
        
        if selected_city not in data:
            st.error(f"No data available for {selected_city}")
            return
        
        df = data[selected_city]
        
        # Create dual-axis plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Over Time', 'Energy Consumption Over Time'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Temperature plot
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['avg_temp_f'],
                mode='lines',
                name='Avg Temperature',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Energy consumption plot
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['energy_consumption_mwh'],
                mode='lines',
                name='Energy Consumption',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Highlight weekends if available
        if 'is_weekend' in df.columns:
            weekend_data = df[df['is_weekend']]
            if len(weekend_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=weekend_data['date'],
                        y=weekend_data['avg_temp_f'],
                        mode='markers',
                        name='Weekend Temp',
                        marker=dict(color='orange', size=6),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=weekend_data['date'],
                        y=weekend_data['energy_consumption_mwh'],
                        mode='markers',
                        name='Weekend Energy',
                        marker=dict(color='green', size=6),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=f"Temperature and Energy Consumption - {selected_city}",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
        fig.update_yaxes(title_text="Energy Consumption (MWh)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_analysis(self, data: Dict[str, pd.DataFrame]):
        """Render correlation analysis visualization."""
        st.subheader("🔗 Correlation Analysis")
        
        if not data:
            st.warning("No data available for correlation analysis")
            return
        
        # Calculate correlations
        try:
            correlation_results = self.analyzer.calculate_correlations(data)
            
            if len(correlation_results) == 0:
                st.warning("No correlation results available")
                return
            
            # Display correlation table
            st.subheader("Correlation Coefficients")
            display_cols = ['city', 'avg_temp_correlation', 'correlation_strength', 'r_squared', 'p_value']
            display_df = correlation_results[display_cols].copy()
            display_df.columns = ['City', 'Correlation', 'Strength', 'R²', 'P-value']
            
            # Style the dataframe
            def style_correlation(val):
                if isinstance(val, (int, float)):
                    if abs(val) >= 0.7:
                        return 'color: green; font-weight: bold'
                    elif abs(val) >= 0.5:
                        return 'color: orange; font-weight: bold'
                    else:
                        return 'color: red'
                return ''
            
            styled_df = display_df.style.applymap(style_correlation, subset=['Correlation'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Scatter plot with regression line
            st.subheader("Temperature vs Energy Consumption")
            
            # Combine all data for scatter plot
            all_data = []
            for city, df in data.items():
                city_data = df[['avg_temp_f', 'energy_consumption_mwh']].copy()
                city_data['city'] = city
                all_data.append(city_data)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                fig = px.scatter(
                    combined_df,
                    x='avg_temp_f',
                    y='energy_consumption_mwh',
                    color='city',
                    title="Temperature vs Energy Consumption by City",
                    labels={
                        'avg_temp_f': 'Average Temperature (°F)',
                        'energy_consumption_mwh': 'Energy Consumption (MWh)'
                    },
                    trendline="ols",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in correlation analysis: {e}")
    
    def render_seasonal_patterns(self, data: Dict[str, pd.DataFrame]):
        """Render seasonal patterns analysis."""
        st.subheader("🍂 Seasonal Patterns")
        
        if not data:
            st.warning("No data available for seasonal analysis")
            return
        
        try:
            seasonal_results = self.analyzer.analyze_seasonal_patterns(data)
            
            if not seasonal_results:
                st.warning("No seasonal analysis results available")
                return
            
            # Create seasonal comparison chart
            seasonal_data = []
            for city, df in seasonal_results.items():
                for _, row in df.iterrows():
                    seasonal_data.append({
                        'city': city,
                        'season': row['season'],
                        'avg_temp': row['avg_temp_f_mean'],
                        'avg_energy': row['energy_consumption_mwh_mean']
                    })
            
            if seasonal_data:
                seasonal_df = pd.DataFrame(seasonal_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_temp = px.bar(
                        seasonal_df,
                        x='season',
                        y='avg_temp',
                        color='city',
                        title="Average Temperature by Season",
                        labels={'avg_temp': 'Average Temperature (°F)'}
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with col2:
                    fig_energy = px.bar(
                        seasonal_df,
                        x='season',
                        y='avg_energy',
                        color='city',
                        title="Average Energy Consumption by Season",
                        labels={'avg_energy': 'Average Energy Consumption (MWh)'}
                    )
                    st.plotly_chart(fig_energy, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error in seasonal analysis: {e}")
    
    def render_weekend_analysis(self, data: Dict[str, pd.DataFrame]):
        """Render weekend vs weekday analysis."""
        st.subheader("📅 Weekend vs Weekday Analysis")
        
        if not data:
            st.warning("No data available for weekend analysis")
            return
        
        try:
            weekend_results = self.analyzer.analyze_weekend_patterns(data)
            
            if len(weekend_results) == 0:
                st.warning("No weekend analysis results available")
                return
            
            # Display weekend comparison table
            display_cols = ['city', 'weekday_avg_mwh', 'weekend_avg_mwh', 'pct_difference', 'weekend_higher']
            display_df = weekend_results[display_cols].copy()
            display_df.columns = ['City', 'Weekday Avg (MWh)', 'Weekend Avg (MWh)', 'Difference (%)', 'Weekend Higher']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                weekend_results,
                x='city',
                y=['weekday_avg_mwh', 'weekend_avg_mwh'],
                title="Weekend vs Weekday Energy Consumption",
                labels={'value': 'Energy Consumption (MWh)', 'variable': 'Day Type'},
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in weekend analysis: {e}")
    
    def render_insights(self, data: Dict[str, pd.DataFrame]):
        """Render business insights."""
        st.subheader("💡 Business Insights")
        
        if not data:
            st.warning("No data available for insights generation")
            return
        
        try:
            analysis_results = self.analyzer.run_full_analysis(data)
            insights = analysis_results.get('insights', {})
            
            if not insights:
                st.warning("No insights generated")
                return
            
            # Display key insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔗 Correlation Insights")
                corr_insights = insights.get('correlations', {})
                if corr_insights:
                    st.write(f"**Cities with strong correlation:** {corr_insights.get('cities_with_strong_correlation', 0)}")
                    st.write(f"**Average correlation:** {corr_insights.get('average_correlation', 0):.3f}")
                    st.write(f"**Strongest correlation:** {corr_insights.get('strongest_correlation_city', 'N/A')}")
            
            with col2:
                st.subheader("📅 Weekend Insights")
                weekend_insights = insights.get('weekend', {})
                if weekend_insights:
                    st.write(f"**Cities with higher weekend usage:** {weekend_insights.get('cities_weekend_higher', 0)}")
                    st.write(f"**Average weekend difference:** {weekend_insights.get('average_weekend_difference_pct', 0):.1f}%")
            
            # Recommendations
            recommendations = insights.get('recommendations', [])
            if recommendations:
                st.subheader("📋 Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            
        except Exception as e:
            st.error(f"Error generating insights: {e}")
    
    def run(self):
        """Run the main dashboard."""
        self.render_header()
        
        # Get sidebar parameters
        params = self.render_sidebar()
        
        if not params['selected_cities']:
            st.warning("Please select at least one city to analyze")
            return
        
        # Load data
        if len(params['date_range']) == 2:
            start_date, end_date = params['date_range']
            with st.spinner("Loading data..."):
                data = self.load_data(start_date.isoformat(), end_date.isoformat(), params['selected_cities'])
        else:
            st.error("Please select a valid date range")
            return
        
        # Render visualizations
        self.render_overview_metrics(data)
        st.markdown("---")
        
        self.render_geographic_overview(data)
        st.markdown("---")
        
        self.render_time_series_analysis(data)
        st.markdown("---")
        
        if params['show_correlations']:
            self.render_correlation_analysis(data)
            st.markdown("---")
        
        if params['show_seasonal']:
            self.render_seasonal_patterns(data)
            st.markdown("---")
        
        if params['show_weekend']:
            self.render_weekend_analysis(data)
            st.markdown("---")
        
        self.render_insights(data)

# Main execution
if __name__ == "__main__":
    dashboard = WeatherEnergyDashboard()
    dashboard.run()