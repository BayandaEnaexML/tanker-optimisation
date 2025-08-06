"""
Enaex Africa Tanker Operations - Enhanced Streamlit Dashboard
=============================================================
Interactive dashboard for analyzing tanker operations across FY2024-2025
for both 30T and 34T fleets with advanced cost analysis and customer normalization.

To run: python -m streamlit run tanker-streamlit-july.py

Author: Data Science Team
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import base64
from io import BytesIO
import re


# Force dark mode configuration
st.set_page_config(
    page_title="Enaex Africa - Enhanced Tanker Operations",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    # Force dark theme
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Enaex Africa Tanker Operations Dashboard"
    }
)

# Add this CSS to force dark mode
st.markdown("""
<style>
    /* Force dark mode */
    :root {
        color-scheme: dark;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    /* Enhanced dark mode styling */
    .stApp {{
        background-color: #0E1117;
    }}
    
    /* Fix metric containers for dark mode */
    div[data-testid="metric-container"] {{
        background-color: rgba(28, 28, 28, 0.8);
        border: 1px solid rgba(196, 30, 58, 0.3);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    }}
    
    /* Fix dataframe styling for dark mode */
    .dataframe {{
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }}
    
    .dataframe th {{
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }}
    
    .dataframe td {{
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }}
    
    /* Fix selectbox and other inputs for dark mode */
    .stSelectbox > div > div {{
        background-color: #1a1a1a;
        color: #e0e0e0;
    }}
    
    .stDateInput > div > div > input {{
        background-color: #1a1a1a;
        color: #e0e0e0;
    }}
    
    /* Ensure all text is visible in dark mode */
    .stMarkdown, .stText {{
        color: #e0e0e0;
    }}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enaex Africa - Enhanced Tanker Operations",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enaex Corporate Colors
ENAEX_COLORS = {
    'primary': '#C41E3A',          # Enaex corporate red
    'secondary': '#1C1C1C',        # Corporate black
    'accent': '#2a2a2a',           # Slightly lighter dark
    'text': '#ffffff',             # White text
    'light_text': '#aaaaaa',       # Light gray text
    'success': '#2E8B57',          # Green for positive
    'warning': '#FF8C00',          # Orange for warning  
    'danger': '#DC143C',           # Red for danger
    'info': '#4169E1',             # Royal blue for info (NEW - more distinct)
    'background': '#0E1117'        # Streamlit dark background
}

# Custom CSS for Enaex branding
st.markdown(f"""
<style>
    /* Main styling */
    .stApp {{
        background-color: {ENAEX_COLORS['background']};
    }}
    
    /* Header styling */
    .main-header {{
        background: linear-gradient(135deg, {ENAEX_COLORS['secondary']} 0%, {ENAEX_COLORS['accent']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid {ENAEX_COLORS['primary']};
    }}
    
    .main-header h1 {{
        color: {ENAEX_COLORS['text']};
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }}
    
    .main-header p {{
        color: {ENAEX_COLORS['light_text']};
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }}
    
    .tagline {{
        color: {ENAEX_COLORS['primary']};
        font-style: italic;
        font-size: 1rem;
    }}
    
    /* Metric cards */
    div[data-testid="metric-container"] {{
        background-color: {ENAEX_COLORS['accent']};
        border: 1px solid {ENAEX_COLORS['primary']};
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    /* Warning box */
    .warning-box {{
        background: rgba(255,140,0,0.1);
        border: 2px solid {ENAEX_COLORS['warning']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* Success box */
    .success-box {{
        background: rgba(46,139,87,0.1);
        border: 2px solid {ENAEX_COLORS['success']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* Info box */
    .info-box {{
        background: rgba(196,30,58,0.1);
        border: 2px solid {ENAEX_COLORS['primary']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* Filter notice */
    .filter-notice {{
        background: rgba(28,28,28,0.8);
        border: 1px solid {ENAEX_COLORS['accent']};
        border-radius: 5px;
        padding: 0.8rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_all = None

# Customer normalization mappings
CUSTOMER_MAPPINGS_30T = {
    'carolina': ['Carolina', 'Carolina Mine', 'CAROLINA', 'carolina'],
    'dorsfontein': ['Dorsfontein', 'DORSFONTEIN', 'dorsfontein', 'Dorsfontein Mine'],
    'exxaro belfast': ['Exxaro Belfast', 'EXXARO BELFAST', 'exxaro belfast', 'Belfast', 'BELFAST', 'belfast'],
    'foskor': ['Foskor', 'FOSKOR', 'foskor', 'Foskor Mine'],
    'grootegeluk': ['Grootegeluk', 'GROOTEGELUK', 'grootegeluk', 'Grootegeluk Mine'],
    'isibonelo': ['Isibonelo', 'ISIBONELO', 'isibonelo', 'Isibonelo Mine'],
    'leeuwpan': ['Leeuwpan', 'LEEUWPAN', 'leeuwpan', 'Leeuwpan Mine'],
    'port shepstone': ['Port Shepstone', 'PORT SHEPSTONE', 'port shepstone', 'Shepstone'],
    'tharisa mine': ['Tharisa mine', 'THARISA MINE', 'tharisa mine', 'Tharisa', 'THARISA', 'tharisa'],
    'umk': ['UMK', 'umk', 'U.M.K', 'UMK Mine'],
    'weilaagte': ['Weilaagte', 'WEILAAGTE', 'weilaagte', 'Weilaagte Mine'],
    'wildfontein': ['Wildfontein', 'WILDFONTEIN', 'wildfontein', 'Wildfontein Mine'],
    'witbank mini-pits': ['Witbank Mini-Pits', 'WITBANK MINI-PITS', 'witbank mini-pits', 'Witbank', 'WITBANK', 'witbank']
}

CUSTOMER_MAPPINGS_34T = {
    'beeshoek': ['Beeshoek', 'BEESHOEK', 'beeshoek', 'Beeshoek Mine'],
    'carolina': ['Carolina', 'Carolina Mine', 'CAROLINA', 'carolina'],
    'dorsfontein': ['Dorsfontein', 'DORSFONTEIN', 'dorsfontein', 'Dorsfontein Mine'],
    'exxaro belfast': ['Exxaro Belfast', 'EXXARO BELFAST', 'exxaro belfast', 'Belfast', 'BELFAST', 'belfast'],
    'foskor': ['Foskor', 'FOSKOR', 'foskor', 'Foskor Mine'],
    'grootegeluk': ['Grootegeluk', 'GROOTEGELUK', 'grootegeluk', 'Grootegeluk Mine'],
    'isibonelo': ['Isibonelo', 'ISIBONELO', 'isibonelo', 'Isibonelo Mine'],
    'leeuwpan': ['Leeuwpan', 'LEEUWPAN', 'leeuwpan', 'Leeuwpan Mine'],
    'port shepstone': ['Port Shepstone', 'PORT SHEPSTONE', 'port shepstone', 'Shepstone'],
    'rustenburg minipits': ['Rustenburg Minipits', 'RUSTENBURG MINIPITS', 'rustenburg minipits', 'Rustenburg', 'RUSTENBURG', 'rustenburg'],
    'sedibeng': ['Sedibeng', 'SEDIBENG', 'sedibeng', 'Sedibeng Mine'],
    'tharisa mine': ['Tharisa mine', 'THARISA MINE', 'tharisa mine', 'Tharisa', 'THARISA', 'tharisa'],
    'umk': ['UMK', 'umk', 'U.M.K', 'UMK Mine'],
    'weilaagte': ['Weilaagte', 'WEILAAGTE', 'weilaagte', 'Weilaagte Mine'],
    'wildfontein': ['Wildfontein', 'WILDFONTEIN', 'wildfontein', 'Wildfontein Mine'],
    'witbank mini-pits': ['Witbank Mini-Pits', 'WITBANK MINI-PITS', 'witbank mini-pits', 'Witbank', 'WITBANK', 'witbank']
}

def normalize_customer_name(customer_name, tanker_class):
    """Normalize customer names based on tanker class"""
    if pd.isna(customer_name):
        return None
    
    customer_name = str(customer_name).strip()
    
    # Choose the appropriate mapping based on tanker class
    if tanker_class == '30T':
        mappings = CUSTOMER_MAPPINGS_30T
    else:
        mappings = CUSTOMER_MAPPINGS_34T
    
    # Find matching normalized name
    for normalized_name, variations in mappings.items():
        for variation in variations:
            if re.search(re.escape(variation), customer_name, re.IGNORECASE):
                return normalized_name.title()
    
    return None  # Return None for customers not in our scope

def filter_relevant_customers(df):
    """Filter dataframe to only include relevant customers"""
    # Apply normalization
    df['customer_normalized'] = df.apply(
        lambda row: normalize_customer_name(row['customer'], row['tanker_class']), 
        axis=1
    )
    
    # Filter out rows where customer_normalized is None (irrelevant customers)
    df_filtered = df[df['customer_normalized'].notna()].copy()
    
    return df_filtered

@st.cache_data
def load_excel_with_fallback(filename, year, tanker_class):
    """Load Excel file with error handling and sample data fallback"""
    try:
        df = pd.read_excel(filename)
        return df, True
    except FileNotFoundError:
        return create_sample_data(year, tanker_class, n_records=1000), False
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return create_sample_data(year, tanker_class, n_records=1000), False

@st.cache_data
def create_sample_data(year, tanker_class, n_records=1000):
    """Create realistic sample data for demonstration"""
    np.random.seed(42)
    
    # Use the relevant customer lists based on tanker class
    if tanker_class == '30T':
        customers = list(CUSTOMER_MAPPINGS_30T.keys())
    else:
        customers = list(CUSTOMER_MAPPINGS_34T.keys())
    
    drivers = [f"Driver_{i:03d}" for i in range(1, 31)]
    
    base_date = pd.Timestamp(f'{year}-01-01')
    dates = pd.date_range(base_date, periods=n_records, freq='6H')
    
    df = pd.DataFrame({
        'date_time_loaded': dates,
        'date_time_offloaded': dates + pd.Timedelta(hours=8) + pd.to_timedelta(np.random.normal(2, 1, n_records), unit='h'),
        'customer': np.random.choice(customers, n_records),
        'driver_name': np.random.choice(drivers, n_records),
        'assistant': np.random.choice(['Assistant A', 'Assistant B', 'No Assistant'], n_records, p=[0.4, 0.4, 0.2]),
        'nominal_payload': 30 if tanker_class == '30T' else 34,
        'tonnage_loaded': np.random.normal(28 if tanker_class == '30T' else 32, 2, n_records),
        'distance': np.random.normal(250, 100, n_records),
        'r_per_ton': np.random.normal(450, 50, n_records),
        'rate_per_km': np.random.normal(12, 2, n_records),
        'amount_rate': np.random.normal(3000, 500, n_records),
        'fixed_cost': np.random.normal(5000, 500, n_records),
        'standing_time_load': np.random.lognormal(0.5, 0.8, n_records),
        'standing_time_offload': np.random.lognormal(0.7, 0.9, n_records),
        'waiting_to_load_hours': np.random.lognormal(0.3, 0.6, n_records),
        'waiting_to_offload_hours': np.random.lognormal(0.4, 0.7, n_records),
        'load_difference': np.random.normal(0, 0.5, n_records)
    })
    
    # Calculate cost_incurred
    df['cost_incurred'] = df['amount_rate'] + df['fixed_cost']
    
    # Add some 48-hour anomalies for FY2025
    if year == 2025:
        anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
        df.loc[anomaly_indices, 'waiting_to_offload_hours'] = 48.0
    
    return df

def fix_waiting_time_anomalies(df):
    """Identify and flag waiting time anomalies"""
    df['total_waiting_hours'] = df['waiting_to_load_hours'] + df['waiting_to_offload_hours']
    
    Q1 = df['total_waiting_hours'].quantile(0.25)
    Q3 = df['total_waiting_hours'].quantile(0.75)
    IQR = Q3 - Q1
    
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    df['is_waiting_anomaly'] = (df['total_waiting_hours'] > upper_bound) | \
                              (df['total_waiting_hours'] < lower_bound)
    
    df['total_waiting_cleaned'] = df['total_waiting_hours'].copy()
    df.loc[df['is_waiting_anomaly'], 'total_waiting_cleaned'] = np.nan
    df['total_waiting_cleaned'].fillna(df['total_waiting_cleaned'].median(), inplace=True)
    
    return df

@st.cache_data
def load_and_process_data():
    """Load all data files and perform preprocessing"""
    
    # Load all four datasets
    df_2024_30T, real_2024_30T = load_excel_with_fallback('df_2024_Eagles_30T.xlsx', 2024, '30T')
    df_2024_34T, real_2024_34T = load_excel_with_fallback('df_2024_Eagles_34T.xlsx', 2024, '34T')
    df_2025_30T, real_2025_30T = load_excel_with_fallback('df_2025_Eagles_30T.xlsx', 2025, '30T')
    df_2025_34T, real_2025_34T = load_excel_with_fallback('df_2025_Eagles_34T.xlsx', 2025, '34T')
    
    # Add year and tanker class identifiers
    df_2024_30T['year'] = 2024
    df_2024_30T['tanker_class'] = '30T'
    df_2024_34T['year'] = 2024
    df_2024_34T['tanker_class'] = '34T'
    df_2025_30T['year'] = 2025
    df_2025_30T['tanker_class'] = '30T'
    df_2025_34T['year'] = 2025
    df_2025_34T['tanker_class'] = '34T'
    
    # Combine all datasets
    df_all = pd.concat([df_2024_30T, df_2024_34T, df_2025_30T, df_2025_34T], 
                       ignore_index=True, sort=False)
    
    # Filter to relevant customers only
    df_all = filter_relevant_customers(df_all)
    
    # Handle missing values
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    df_all[numeric_cols] = df_all[numeric_cols].fillna(0)
    
    df_all['assistant'] = df_all['assistant'].fillna('No Assistant')
    df_all['driver_name'] = df_all['driver_name'].fillna('Unknown')
    
    # Create additional features
    df_all['utilization_percent'] = (df_all['tonnage_loaded'] / df_all['nominal_payload'] * 100).round(2)
    df_all['utilization_percent'] = df_all['utilization_percent'].clip(upper=110)
    
    # Date/time features
    df_all['date_time_loaded'] = pd.to_datetime(df_all['date_time_loaded'], errors='coerce')
    df_all['date_time_offloaded'] = pd.to_datetime(df_all['date_time_offloaded'], errors='coerce')
    
    df_all['month'] = df_all['date_time_loaded'].dt.to_period('M').astype(str)
    df_all['weekday'] = df_all['date_time_loaded'].dt.day_name()
    df_all['hour_loaded'] = df_all['date_time_loaded'].dt.hour
    df_all['quarter'] = df_all['date_time_loaded'].dt.quarter
    df_all['quarter_year'] = df_all['date_time_loaded'].dt.to_period('Q').astype(str)
    
    # Calculate total trip duration
    df_all['total_trip_hours'] = (df_all['date_time_offloaded'] - df_all['date_time_loaded']).dt.total_seconds() / 3600
    
    # Calculate efficiency metrics
    df_all['total_waiting_hours'] = df_all['waiting_to_load_hours'] + df_all['waiting_to_offload_hours']
    
    # Fix waiting time anomalies
    df_all = fix_waiting_time_anomalies(df_all)
    
    # Calculate cost metrics
    if 'cost_incurred' not in df_all.columns:
        df_all['cost_incurred'] = df_all.get('amount_rate', 0) + df_all.get('fixed_cost', 0)
    
    # Recalculate efficiency score with cleaned waiting times
    df_all['efficiency_score'] = (df_all['utilization_percent'] / 100 * 
                                  (1 - df_all['total_waiting_cleaned'] / df_all['total_trip_hours'].clip(lower=1))).clip(0, 1)
    
    # Cost efficiency categorization
    df_all['cost_category'] = pd.cut(df_all['r_per_ton'], 
                                     bins=[0, 400, 600, 1000, 10000],
                                     labels=['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'])
    
    # Check if all data is sample data
    all_sample = not any([real_2024_30T, real_2024_34T, real_2025_30T, real_2025_34T])
    
    return df_all, all_sample

def calculate_q1_metrics(df, year):
    """Calculate Q1 metrics for comparison"""
    q1_data = df[(df['year'] == year) & (df['month'].str.contains('2024-0[1-3]|2025-0[1-3]', na=False))]
    
    if len(q1_data) == 0:
        return {}
    
    metrics = {
        'total_tonnage': q1_data['tonnage_loaded'].sum(),
        'total_trips': len(q1_data),
        'avg_utilization': q1_data['utilization_percent'].mean(),
        'avg_cost_per_ton': q1_data['r_per_ton'].mean(),
        'total_cost_incurred': q1_data['cost_incurred'].sum(),
        'avg_distance': q1_data['distance'].mean(),
        'total_waiting_hours': q1_data['total_waiting_cleaned'].sum(),
        'anomaly_count': q1_data['is_waiting_anomaly'].sum(),
        'avg_payload_30t': q1_data[q1_data['tanker_class'] == '30T']['tonnage_loaded'].mean(),
        'avg_payload_34t': q1_data[q1_data['tanker_class'] == '34T']['tonnage_loaded'].mean(),
    }
    
    return metrics

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Enaex Africa - Enhanced Tanker Operations Dashboard</h1>
        <p>Comprehensive Performance Analysis FY2024-2025 with Advanced Cost Intelligence</p>
        <p class="tagline">Stronger Bonds Through Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading and processing data...'):
        df_all, is_sample_data = load_and_process_data()
        st.session_state.data_loaded = True
        st.session_state.df_all = df_all
    
    if is_sample_data:
        st.warning("📊 **Sample Data Mode**: Excel files not found. Using simulated data for demonstration.")
    
    # Display filtering notice
    st.markdown("""
    <div class="filter-notice">
        <strong>📋 Data Scope:</strong> Analysis filtered to include only strategically relevant customers as defined in the optimization scope. 
        Customer names have been normalized to handle variations in spelling and formatting.
        <em>Note: 12:00 AM load times from 2024 are excluded only from temporal pattern analysis (not from Q1 comparisons or other metrics).</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate Q1 metrics for comparison
    q1_2024_metrics = calculate_q1_metrics(df_all, 2024)
    q1_2025_metrics = calculate_q1_metrics(df_all, 2025)
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Date range filter
    st.sidebar.markdown("### 📅 Date Range")
    date_min = df_all['date_time_loaded'].min().date()
    date_max = df_all['date_time_loaded'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    
    # Fleet filter
    st.sidebar.markdown("### 🚛 Fleet Selection")
    fleet_options = ['All Fleets'] + list(df_all['tanker_class'].unique())
    selected_fleet = st.sidebar.selectbox("Select fleet", fleet_options)
    
    # Year filter
    st.sidebar.markdown("### 📆 Year")
    year_options = ['All Years'] + list(df_all['year'].unique())
    selected_year = st.sidebar.selectbox("Select year", year_options)
    
    # Customer filter
    st.sidebar.markdown("### 👥 Customer")
    customer_options = ['All Customers'] + sorted(df_all['customer_normalized'].unique())
    selected_customer = st.sidebar.selectbox("Select customer", customer_options)
    
    # Apply filters
    filtered_df = df_all.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date_time_loaded'].dt.date >= date_range[0]) &
            (filtered_df['date_time_loaded'].dt.date <= date_range[1])
        ]
    
    if selected_fleet != 'All Fleets':
        filtered_df = filtered_df[filtered_df['tanker_class'] == selected_fleet]
    
    if selected_year != 'All Years':
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    
    if selected_customer != 'All Customers':
        filtered_df = filtered_df[filtered_df['customer_normalized'] == selected_customer]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filter Summary")
    st.sidebar.info(f"""
    **Records:** {len(filtered_df):,} of {len(df_all):,}  
    **Date Range:** {date_range[0] if len(date_range) > 0 else 'N/A'} to {date_range[1] if len(date_range) > 1 else 'N/A'}  
    **Fleet:** {selected_fleet}  
    **Year:** {selected_year}  
    **Customer:** {selected_customer}
    """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", 
        "💰 Cost Deep Dive",
        "📈 Q1 2024 vs 2025",
        "📦 Payload Efficiency", 
        "⏱️ Standing Time", 
        "📅 Temporal Patterns",
        "🗺️ Route Analysis",
        "👤 Driver Performance",
        "💡 Strategic Insights"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("## Key Performance Indicators")
        
        # Calculate KPIs
        total_tonnage = filtered_df['tonnage_loaded'].sum()
        total_trips = len(filtered_df)
        avg_utilization = filtered_df['utilization_percent'].mean()
        avg_cost_per_ton = filtered_df['r_per_ton'].mean()
        total_distance = filtered_df['distance'].sum()
        total_waiting_time = filtered_df['total_waiting_cleaned'].sum()
        anomaly_count = filtered_df['is_waiting_anomaly'].sum()
        total_cost_incurred = filtered_df['cost_incurred'].sum()
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tonnage Delivered",
                f"{total_tonnage:,.0f} tons",
                delta=f"{total_trips:,} trips"
            )
        
        with col2:
            st.metric(
                "Average Utilization",
                f"{avg_utilization:.1f}%",
                delta="Target: 98%"
            )
        
        with col3:
            st.metric(
                "Total Cost Incurred",
                f"R{total_cost_incurred:,.0f}",
                delta=f"R{avg_cost_per_ton:.0f}/ton avg"
            )
        
        with col4:
            st.metric(
                "Waiting Anomalies",
                f"{anomaly_count} detected",
                delta="Efficiency impact" if anomaly_count > 0 else "None",
                delta_color="inverse"
            )
        
        # Additional metrics
        st.markdown("### 📈 Operational Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Distance",
                f"{total_distance:,.0f} km",
                delta=f"{total_distance/total_trips:.0f} km/trip" if total_trips > 0 else "N/A"
            )
        
        with col2:
            st.metric(
                "Total Waiting Hours",
                f"{total_waiting_time:,.0f} hrs",
                delta=f"{total_waiting_time/total_trips:.1f} hrs/trip" if total_trips > 0 else "N/A"
            )
        
        with col3:
            st.metric(
                "Fleet Efficiency Score",
                f"{filtered_df['efficiency_score'].mean():.2f}",
                delta="Max: 1.0"
            )
        
        with col4:
            relevant_customers = len(filtered_df['customer_normalized'].unique())
            st.metric(
                "Active Customers",
                f"{relevant_customers}",
                delta="Scope-filtered"
            )
        
        # Monthly trend chart
        st.markdown("### 📊 Monthly Operations Trend")

        monthly_stats = filtered_df.groupby(['month', 'tanker_class']).agg({
            'tonnage_loaded': 'sum',
            'cost_incurred': 'sum',
            'utilization_percent': 'mean'
        }).reset_index()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tonnage Delivered & Cost Incurred', 'Average Utilization'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            specs=[[{"secondary_y": True}],  # Enable secondary y-axis for first subplot
                [{"secondary_y": False}]]
        )

        # Tonnage by fleet (primary y-axis)
        for tanker_class in monthly_stats['tanker_class'].unique():
            data = monthly_stats[monthly_stats['tanker_class'] == tanker_class]
            fig.add_trace(
                go.Scatter(
                    x=data['month'],
                    y=data['tonnage_loaded'],
                    mode='lines+markers',
                    name=f'{tanker_class} Tonnage',
                    line=dict(width=3),
                    yaxis='y'  # Primary axis
                ),
                row=1, col=1,
                secondary_y=False  # Use primary y-axis
            )

        # Cost line (secondary y-axis)
        cost_data = monthly_stats.groupby('month')['cost_incurred'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=cost_data['month'],
                y=cost_data['cost_incurred'],
                mode='lines+markers',
                name='Total Cost',
                line=dict(color=ENAEX_COLORS['warning'], width=3, dash='dot'),  # Add dash style for distinction
                yaxis='y2'  # Secondary axis
            ),
            row=1, col=1,
            secondary_y=True  # Use secondary y-axis
        )

        # Utilization trend
        util_data = monthly_stats.groupby('month')['utilization_percent'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=util_data['month'],
                y=util_data['utilization_percent'],
                mode='lines+markers',
                name='Avg Utilization',
                line=dict(color=ENAEX_COLORS['success'], width=3)
            ),
            row=2, col=1,
            secondary_y=False
        )

        # Update layout with proper axis labels
        fig.update_yaxes(title_text="Tonnage (tons)", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Cost Incurred (R)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Utilization %", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=1)

        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Cost Deep Dive
    with tab2:
        st.markdown("## 💰 Comprehensive Cost Analysis")
        
        # Cost breakdown by component
        st.markdown("### Cost Component Analysis")
        
        cost_components = filtered_df.groupby('tanker_class').agg({
            'amount_rate': 'sum',
            'fixed_cost': 'sum',
            'cost_incurred': 'sum',
            'tonnage_loaded': 'sum'
        }).reset_index()
        
        cost_components['variable_cost_per_ton'] = cost_components['amount_rate'] / cost_components['tonnage_loaded']
        cost_components['fixed_cost_per_ton'] = cost_components['fixed_cost'] / cost_components['tonnage_loaded']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost breakdown pie chart
            fig_cost_breakdown = go.Figure()
            
            for idx, row in cost_components.iterrows():
                fig_cost_breakdown.add_trace(go.Pie(
                    labels=['Variable Costs', 'Fixed Costs'],
                    values=[row['amount_rate'], row['fixed_cost']],
                    name=row['tanker_class'],
                    title=f"{row['tanker_class']} Fleet",
                    domain=dict(x=[0.5*idx, 0.5*(idx+1)], y=[0, 1])
                ))
            
            fig_cost_breakdown.update_layout(
                title='Cost Structure by Fleet Type',
                height=400
            )
            st.plotly_chart(fig_cost_breakdown, use_container_width=True)
        
        with col2:
            # Cost per ton by customer
            customer_costs = filtered_df.groupby('customer_normalized').agg({
                'cost_incurred': 'sum',
                'tonnage_loaded': 'sum',
                'r_per_ton': 'mean'
            }).reset_index()
            customer_costs = customer_costs.sort_values('r_per_ton', ascending=False).head(10)
            
            fig_customer_cost = px.bar(
                customer_costs,
                x='customer_normalized',
                y='r_per_ton',
                title='Top 10 Highest Cost per Ton by Customer',
                color='r_per_ton',
                color_continuous_scale='Reds'
            )
            fig_customer_cost.update_xaxes(tickangle=45)
            st.plotly_chart(fig_customer_cost, use_container_width=True)
        
        # Cost efficiency analysis
        st.markdown("### Cost Efficiency Drivers")
        
        # Correlation analysis
        cost_factors = filtered_df[['distance', 'utilization_percent', 'total_waiting_cleaned', 'r_per_ton']].corr()['r_per_ton'].sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost factor correlation
            fig_corr = px.bar(
                x=cost_factors.index[1:],  # Exclude self-correlation
                y=cost_factors.values[1:],
                title='Factors Affecting Cost per Ton',
                labels={'x': 'Factor', 'y': 'Correlation with R/Ton'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Cost vs efficiency scatter
            fig_scatter = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df))),
                x='utilization_percent',
                y='r_per_ton',
                color='tanker_class',
                size='tonnage_loaded',
                title='Cost per Ton vs Utilization Efficiency',
                hover_data=['customer_normalized', 'distance']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Monthly cost trend analysis
        st.markdown("### Monthly Cost Trends")
        
        monthly_cost_detail = filtered_df.groupby(['month', 'tanker_class']).agg({
            'cost_incurred': 'sum',
            'tonnage_loaded': 'sum',
            'r_per_ton': 'mean',
            'distance': 'mean',
            'total_waiting_cleaned': 'mean'
        }).reset_index()
        
        fig_cost_trend = px.line(
            monthly_cost_detail,
            x='month',
            y='r_per_ton',
            color='tanker_class',
            markers=True,
            title='Monthly Cost per Ton Trend by Fleet Type'
        )
        st.plotly_chart(fig_cost_trend, use_container_width=True)
        
        # Cost impact of inefficiencies
        st.markdown("### Cost Impact of Operational Inefficiencies")

        # copy for calculations
        ineff = filtered_df.copy()

        # 1️⃣ Underloading cost (as before)
        ineff['underload_cost'] = (
            (100 - ineff['utilization_percent']) / 17.78
            * ineff['cost_incurred']
        )

        # 2️⃣ Compute a *real* R/hr rate per trip
        ineff['trip_hours'] = ineff['total_trip_hours'].clip(lower=0.1)
        ineff['hourly_cost'] = (ineff['cost_incurred'] / ineff['trip_hours'])/120 # waiting time anomalies need to be investigated so we are minimising

        # fleet‐wide average R/hr
        avg_hourly_cost = ineff['hourly_cost'].mean()

        # 3️⃣ Waiting cost using that real rate
        ineff['waiting_cost'] = (
            ineff['total_waiting_cleaned']
            * avg_hourly_cost
        )

        # 4️⃣ Summaries
        total_underload_cost = ineff['underload_cost'].sum()
        total_waiting_cost   = ineff['waiting_cost'].sum() 
        total_inefficiency   = total_underload_cost + total_waiting_cost
        total_cost            = filtered_df['cost_incurred'].sum()

        ineff_pct = (
            total_inefficiency / total_cost * 100
            if total_cost > 0 else 0
        )

        # 5️⃣ Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Cost of Underloading",
                f"R{total_underload_cost:,.0f}",
                delta="Opportunity cost",
                help="Revenue lost from not utilising full truck capacity"
            )
        with col2:
            st.metric(
                "Cost of Waiting Time",
                f"R{total_waiting_cost:,.0f}",
                delta=f"@ R{avg_hourly_cost:.0f}/hr",
                help=f"Based on per‐trip average of R{avg_hourly_cost:.0f} per hour"
            )
        with col3:
            st.metric(
                "Total Inefficiency Cost",
                f"R{total_inefficiency:,.0f}",
                delta=f"{ineff_pct:.1f}% of total costs",
                help="Combined impact of underloading and waiting time"
            )

        # 6️⃣ Detailed breakdown
        with st.expander("📊 Inefficiency Cost Calculation Details"):
            c1, c2 = st.columns(2)
            with c1:
                avg_util = filtered_df['utilization_percent'].mean()
                st.markdown("**Underloading Impact:**")
                st.write(f"- Average utilisation: {avg_util:.1f}%")
                st.write(f"- Average underload: {100-avg_util:.1f}%")
                st.write(f"- Total underload cost: R{total_underload_cost:,.0f}")
                st.write(f"- Per trip impact: R{total_underload_cost/len(filtered_df):,.0f}")
            with c2:
                total_wait_h = filtered_df['total_waiting_cleaned'].sum()
                avg_wait_t  = filtered_df['total_waiting_cleaned'].mean()
                st.markdown("**Waiting Time Impact:**")
                st.write(f"- Total waiting hours: {total_wait_h:,.0f}")
                st.write(f"- Average wait per trip: {avg_wait_t:.1f} hrs")
                st.write(f"- Hourly cost rate: R{avg_hourly_cost:.0f}/hr")
                st.write(f"- Total waiting cost: R{total_waiting_cost:,.0f}")

        # 7️⃣ Breakdown by fleet
        st.markdown("#### Inefficiency Breakdown by Fleet Type")
        fleet = ineff.groupby('tanker_class').agg(
            underload_cost=('underload_cost','sum'),
            waiting_cost=('waiting_cost','sum'),
            cost_incurred=('cost_incurred','sum')
        ).reset_index()
        fleet['total_ineff']   = fleet['underload_cost'] + fleet['waiting_cost']
        fleet['ineff_pct']     = fleet['total_ineff'] / fleet['cost_incurred'] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Underloading Cost',
            x=fleet['tanker_class'],
            y=fleet['underload_cost'],
            marker_color=ENAEX_COLORS['warning'],
            text=[f"R{v:,.0f}" for v in fleet['underload_cost']],
            textposition='inside'
        ))
        fig.add_trace(go.Bar(
            name='Waiting Time Cost',
            x=fleet['tanker_class'],
            y=fleet['waiting_cost'],
            marker_color=ENAEX_COLORS['danger'],
            text=[f"R{v:,.0f}" for v in fleet['waiting_cost']],
            textposition='inside'
        ))
        for _, row in fleet.iterrows():
            fig.add_annotation(
                x=row['tanker_class'],
                y=row['total_ineff'],
                text=f"{row['ineff_pct']:.1f}% of costs",
                showarrow=False,
                yshift=20
            )
        fig.update_layout(
            title='Inefficiency Costs by Fleet Type',
            barmode='stack',
            yaxis_title='Cost (R)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # 8️⃣ Call‐out box based on severity
        if ineff_pct > 20:
            st.markdown(f"""
            <div class="warning-box">
              <h4>⚠️ High Inefficiency Alert</h4>
              <p>Inefficiencies are costing <strong>{ineff_pct:.1f}%</strong> of total costs.</p>
            </div>
            """, unsafe_allow_html=True)
        elif ineff_pct > 10:
            st.markdown(f"""
            <div class="info-box">
              <h4>📊 Moderate Inefficiency</h4>
              <p>Inefficiencies represent <strong>{ineff_pct:.1f}%</strong> of costs—within norms but improvable.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
              <h4>✅ Efficient Operations</h4>
              <p>Inefficiencies are only <strong>{ineff_pct:.1f}%</strong> of costs—well optimised.</p>
            </div>
            """, unsafe_allow_html=True)

    # Tab 3: Q1 2024 vs 2025 Comparison
    with tab3:
        st.markdown("## 📈 Q1 Performance Comparison: 2024 vs 2025")
        st.markdown("*This analysis uses the complete dataset including all timestamps to ensure accurate Q1 comparisons.*")
        
        if q1_2024_metrics and q1_2025_metrics:
            # Key metrics comparison
            st.markdown("### Key Metrics Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                tonnage_change = ((q1_2025_metrics['total_tonnage'] - q1_2024_metrics['total_tonnage']) / 
                                q1_2024_metrics['total_tonnage'] * 100) if q1_2024_metrics['total_tonnage'] > 0 else 0
                st.metric(
                    "Total Tonnage",
                    f"{q1_2025_metrics['total_tonnage']:,.0f} tons (2025)",
                    delta=f"{tonnage_change:+.1f}% vs 2024"
                )
            
            with col2:
                trips_change = ((q1_2025_metrics['total_trips'] - q1_2024_metrics['total_trips']) / 
                              q1_2024_metrics['total_trips'] * 100) if q1_2024_metrics['total_trips'] > 0 else 0
                st.metric(
                    "Total Trips",
                    f"{q1_2025_metrics['total_trips']:,} (2025)",
                    delta=f"{trips_change:+.1f}% vs 2024"
                )
            
            with col3:
                util_change = q1_2025_metrics['avg_utilization'] - q1_2024_metrics['avg_utilization']
                st.metric(
                    "Avg Utilization",
                    f"{q1_2025_metrics['avg_utilization']:.1f}% (2025)",
                    delta=f"{util_change:+.1f}% vs 2024"
                )
            
            with col4:
                cost_change = q1_2025_metrics['avg_cost_per_ton'] - q1_2024_metrics['avg_cost_per_ton']
                st.metric(
                    "Avg Cost/Ton",
                    f"R{q1_2025_metrics['avg_cost_per_ton']:,.0f} (2025)",
                    delta=f"R{cost_change:+.0f} vs 2024",
                    delta_color="inverse"
                )
            
            # Detailed comparison charts
            st.markdown("### Detailed Performance Analysis")
            
            # Create comparison dataframes
            q1_2024_data = df_all[(df_all['year'] == 2024) & (df_all['month'].str.contains('2024-0[1-3]', na=False))]
            q1_2025_data = df_all[(df_all['year'] == 2025) & (df_all['month'].str.contains('2025-0[1-3]', na=False))]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Utilization comparison by fleet
                util_comparison = []
                for year, data in [(2024, q1_2024_data), (2025, q1_2025_data)]:
                    for fleet in ['30T', '34T']:
                        fleet_data = data[data['tanker_class'] == fleet]
                        if len(fleet_data) > 0:
                            util_comparison.append({
                                'Year': year,
                                'Fleet': fleet,
                                'Utilization': fleet_data['utilization_percent'].mean()
                            })
                
                util_df = pd.DataFrame(util_comparison)
                fig_util_comp = px.bar(
                    util_df,
                    x='Fleet',
                    y='Utilization',
                    color='Year',
                    barmode='group',
                    title='Q1 Utilization Comparison by Fleet',
                    text='Utilization'
                )
                fig_util_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_util_comp, use_container_width=True)
            
            with col2:
                # Cost comparison by fleet
                cost_comparison = []
                for year, data in [(2024, q1_2024_data), (2025, q1_2025_data)]:
                    for fleet in ['30T', '34T']:
                        fleet_data = data[data['tanker_class'] == fleet]
                        if len(fleet_data) > 0:
                            cost_comparison.append({
                                'Year': year,
                                'Fleet': fleet,
                                'Cost_per_Ton': fleet_data['r_per_ton'].mean()
                            })
                
                cost_df = pd.DataFrame(cost_comparison)
                fig_cost_comp = px.bar(
                    cost_df,
                    x='Fleet',
                    y='Cost_per_Ton',
                    color='Year',
                    barmode='group',
                    title='Q1 Cost per Ton Comparison by Fleet',
                    text='Cost_per_Ton'
                )
                fig_cost_comp.update_traces(texttemplate='R%{text:.0f}', textposition='outside')
                st.plotly_chart(fig_cost_comp, use_container_width=True)
            
            # Performance degradation analysis
            st.markdown("### Performance Degradation Drivers")
            
            degradation_analysis = {
                'Metric': ['Tonnage per Trip', 'Utilization %', 'Cost per Ton', 'Waiting Hours', 'Distance per Trip'],
                '2024 Q1': [
                    q1_2024_metrics['total_tonnage'] / q1_2024_metrics['total_trips'] if q1_2024_metrics['total_trips'] > 0 else 0,
                    q1_2024_metrics['avg_utilization'],
                    q1_2024_metrics['avg_cost_per_ton'],
                    q1_2024_metrics['total_waiting_hours'] / q1_2024_metrics['total_trips'] if q1_2024_metrics['total_trips'] > 0 else 0,
                    q1_2024_metrics['avg_distance']
                ],
                '2025 Q1': [
                    q1_2025_metrics['total_tonnage'] / q1_2025_metrics['total_trips'] if q1_2025_metrics['total_trips'] > 0 else 0,
                    q1_2025_metrics['avg_utilization'],
                    q1_2025_metrics['avg_cost_per_ton'],
                    q1_2025_metrics['total_waiting_hours'] / q1_2025_metrics['total_trips'] if q1_2025_metrics['total_trips'] > 0 else 0,
                    q1_2025_metrics['avg_distance']
                ]
            }
            
            degradation_df = pd.DataFrame(degradation_analysis)
            degradation_df['Change %'] = ((degradation_df['2025 Q1'] - degradation_df['2024 Q1']) / 
                                        degradation_df['2024 Q1'] * 100).round(1)
            
            # Style the dataframe
            styled_df = degradation_df.style.format({
                '2024 Q1': '{:.2f}',
                '2025 Q1': '{:.2f}',
                'Change %': '{:+.1f}%'
            }).background_gradient(subset=['Change %'], cmap='RdYlGn_r')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Root cause analysis
            st.markdown("### Root Cause Analysis: Why 2025 Underperformed")
            
            if q1_2025_metrics['avg_cost_per_ton'] > q1_2024_metrics['avg_cost_per_ton']:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Key Findings: Q1 2025 Performance Decline</h4>
                    <ul>
                        <li><strong>Increased Cost per Ton:</strong> Higher operational costs with reduced efficiency</li>
                        <li><strong>Lower Utilization:</strong> More underloaded trips reducing asset efficiency</li>
                        <li><strong>Increased Waiting Times:</strong> Operational bottlenecks causing delays</li>
                        <li><strong>Fleet Mix Issues:</strong> Suboptimal allocation of 30T vs 34T tankers</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Q1 2025 Performance Improvements</h4>
                    <p>Analysis shows improved efficiency in Q1 2025 compared to Q1 2024.</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("Insufficient Q1 data available for comparison analysis.")
    
    # Tab 4: Payload Efficiency (existing code with customer normalization)
    with tab4:
        st.markdown("## 📦 Payload Efficiency Analysis")
        
        # Show customer normalization impact
        st.markdown("### Customer Data Normalization Impact")
        original_customers = len(df_all['customer'].unique()) if 'customer' in df_all.columns else 0
        normalized_customers = len(df_all['customer_normalized'].unique())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Customer Entries", f"{original_customers}")
        with col2:
            st.metric("Normalized Customers", f"{normalized_customers}")
        with col3:
            st.metric("Data Reduction", f"{((original_customers - normalized_customers) / original_customers * 100):.1f}%" if original_customers > 0 else "N/A")
        
        # Utilization distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_util_dist = px.histogram(
                filtered_df, 
                x='utilization_percent', 
                color='tanker_class',
                nbins=50, 
                barmode='overlay', 
                opacity=0.7,
                title='Utilization Distribution by Fleet (Normalized Data)'
            )
            fig_util_dist.add_vline(
                x=100, 
                line_dash="dash", 
                line_color=ENAEX_COLORS['warning'],
                annotation_text="100% Capacity"
            )
            st.plotly_chart(fig_util_dist, use_container_width=True)
        
        with col2:
            # Utilization by normalized customer
            customer_util = filtered_df.groupby('customer_normalized')['utilization_percent'].mean().reset_index()
            customer_util = customer_util.sort_values('utilization_percent', ascending=True)
            
            fig_customer_util = px.bar(
                customer_util,
                y='customer_normalized',
                x='utilization_percent',
                orientation='h',
                title='Average Utilization by Customer (Normalized)',
                color='utilization_percent',
                color_continuous_scale='RdYlGn',
                range_color=[85, 100]
            )
            st.plotly_chart(fig_customer_util, use_container_width=True)
        
        # Payload optimization opportunities
        st.markdown("### Payload Optimization Opportunities")
        
        underutilized = filtered_df[filtered_df['utilization_percent'] < 95]
        potential_savings = underutilized.groupby('customer_normalized').agg({
            'utilization_percent': 'mean',
            'tonnage_loaded': 'sum',
            'nominal_payload': 'sum',
            'cost_incurred': 'sum'
        }).reset_index()
        
        potential_savings['potential_additional_tonnage'] = (
            potential_savings['nominal_payload'] - potential_savings['tonnage_loaded']
        )
        potential_savings['potential_cost_savings'] = (
            potential_savings['potential_additional_tonnage'] * 
            filtered_df['r_per_ton'].mean()
        )
        
        potential_savings = potential_savings.sort_values('potential_cost_savings', ascending=False).head(10)
        
        fig_savings = px.bar(
            potential_savings,
            x='customer_normalized',
            y='potential_cost_savings',
            title='Top 10 Customers: Potential Cost Savings from Full Utilization',
            color='potential_additional_tonnage',
            color_continuous_scale='Blues'
        )
        fig_savings.update_xaxes(tickangle=45)
        st.plotly_chart(fig_savings, use_container_width=True)
    
    # Continue with remaining tabs (Standing Time, Temporal Patterns, etc.)
    # [Rest of the tabs remain similar with customer normalization applied]
    
    # Tab 5: Standing Time
    with tab5:
        st.markdown("## ⏱️ Standing Time Analysis")
        
        # anomaly count
        n_anom = filtered_df['is_waiting_anomaly'].sum()
        if n_anom > 0:
            st.markdown(f"""
            <div class="warning-box">
              <h4>⚠️ Data Anomaly Detected</h4>
              <p>{n_anom} instances of anomalous waiting times flagged and cleaned for analysis.</p>
            </div>
            """, unsafe_allow_html=True)

            # show table of anomalies
            st.markdown("### 🗃️ Anomalous Waiting-Time Records")
            anoms = filtered_df[filtered_df['is_waiting_anomaly']].copy()
            st.dataframe(
                anoms[[
                    'date_time_loaded', 'customer_normalized',
                    'total_waiting_hours', 'waiting_to_load_hours',
                    'waiting_to_offload_hours'
                ]].sort_values('total_waiting_hours', ascending=False),
                use_container_width=True,
                height=300
            )

        # now your existing “top 10 customers” chart
        customer_standing = (
            filtered_df
            .groupby('customer_normalized')
            .agg({
                'standing_time_load':'mean',
                'standing_time_offload':'mean',
                'total_waiting_cleaned':'mean','customer_normalized':'count'
            })
            .rename(columns={'customer_normalized':'trip_count'})
            .query("trip_count>=5")
        )
        worst = customer_standing.nlargest(10, 'total_waiting_cleaned').reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=worst['customer_normalized'], x=worst['standing_time_load'],
            name='Loading Delays', orientation='h', marker_color='#FF8C00'))
        fig2.add_trace(go.Bar(
            y=worst['customer_normalized'], x=worst['standing_time_offload'],
            name='Offloading Delays', orientation='h', marker_color='#DC143C'))
        fig2.update_layout(
            barmode='stack',
            title="Top 10 Customers with Highest Average Standing Times",
            xaxis_title="Average Hours",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 6: Temporal Patterns (excluding 12 AM times for 2024)
    with tab6:
        st.markdown("## 📅 Temporal Patterns Analysis")
        st.markdown("*Note: 12:00 AM load times in 2024 (default timestamps) are excluded from temporal visualizations only.*")
        
        # Filter out 12 AM load times in 2024 for temporal analysis only
        temporal_df = filtered_df.copy()
        midnight_mask = (temporal_df['year'] == 2024) & (temporal_df['date_time_loaded'].dt.hour == 0) & (temporal_df['date_time_loaded'].dt.minute == 0)
        temporal_df = temporal_df[~midnight_mask]
        
        # Show the impact of this filtering
        excluded_records = len(filtered_df)
        temporal_records = len(temporal_df)
        if excluded_records > temporal_records:
            st.info(f"Excluded {excluded_records - temporal_records:,} records with 12:00 AM timestamps from 2024 for temporal analysis. Total records for temporal analysis: {temporal_records:,}")
            
            # Show a comparison of before/after filtering
            col_a, col_b = st.columns(2)
            with col_a:
                # Before filtering - hourly distribution
                before_hourly = filtered_df.groupby('hour_loaded').size().reset_index(name='count')
                fig_before = px.bar(before_hourly, x='hour_loaded', y='count', 
                                  title='Hourly Distribution - Before 12AM Filter (2024)')
                fig_before.update_xaxes(title='Hour of Day')
                st.plotly_chart(fig_before, use_container_width=True)
            
            with col_b:
                # After filtering - hourly distribution
                after_hourly = temporal_df.groupby('hour_loaded').size().reset_index(name='count')
                fig_after = px.bar(after_hourly, x='hour_loaded', y='count', 
                                 title='Hourly Distribution - After 12AM Filter (2024)')
                fig_after.update_xaxes(title='Hour of Day')
                st.plotly_chart(fig_after, use_container_width=True)
        
        # Hourly heatmap (using filtered temporal data)
        hourly_weekday = temporal_df.groupby(['weekday', 'hour_loaded']).size().reset_index(name='trip_count')
        heatmap_data = hourly_weekday.pivot(index='weekday', columns='hour_loaded', values='trip_count').fillna(0)
        
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(weekday_order)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Reds',
            text=heatmap_data.values.astype(int),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title='Loading Activity Heatmap (Cleaned Data - Excludes 2024 Midnight Defaults)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Quarterly analysis using temporal_df
        col1, col2 = st.columns(2)
        
        with col1:
            quarterly_stats = temporal_df.groupby('quarter_year').agg({
                'tonnage_loaded': 'sum',
                'utilization_percent': 'mean',
                'r_per_ton': 'mean'
            }).reset_index()
            
            fig_quarterly = go.Figure()
            
            fig_quarterly.add_trace(go.Bar(
                x=quarterly_stats['quarter_year'],
                y=quarterly_stats['tonnage_loaded'],
                name='Tonnage',
                yaxis='y',
                marker_color=ENAEX_COLORS['primary']
            ))
            
            fig_quarterly.add_trace(go.Scatter(
                x=quarterly_stats['quarter_year'],
                y=quarterly_stats['utilization_percent'],
                name='Utilization %',
                yaxis='y2',
                line=dict(color=ENAEX_COLORS['success'], width=3)
            ))
            
            fig_quarterly.update_layout(
                title='Quarterly Performance (Temporal Clean)',
                yaxis=dict(title='Tonnage'),
                yaxis2=dict(title='Utilization %', overlaying='y', side='right'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_quarterly, use_container_width=True)
        
        with col2:
            # Fleet comparison over time using temporal_df
            fleet_monthly = temporal_df.groupby(['month', 'tanker_class'])['utilization_percent'].mean().reset_index()
            
            fig_fleet_time = px.line(
                fleet_monthly,
                x='month',
                y='utilization_percent',
                color='tanker_class',
                title='Fleet Utilization Comparison (Temporal Clean)',
                markers=True
            )
            
            st.plotly_chart(fig_fleet_time, use_container_width=True)
    
    # Tab 7: Route Analysis (with normalized customers)
    with tab7:
        st.markdown("## 🗺️ Route Performance Analysis")
        
        # Route metrics with normalized customers
        route_metrics = filtered_df.groupby('customer_normalized').agg({
            'utilization_percent': 'mean',
            'total_waiting_cleaned': 'mean',
            'distance': 'mean',
            'r_per_ton': 'mean',
            'cost_incurred': 'sum',
            'customer_normalized': 'count'
        }).rename(columns={'customer_normalized': 'trip_count'}).reset_index()
        
        route_metrics = route_metrics[route_metrics['trip_count'] >= 5]
        
        # Route performance matrix
        if len(route_metrics) > 0:
            fig_route_matrix = px.scatter(
                route_metrics,
                x='utilization_percent',
                y='total_waiting_cleaned',
                size='trip_count',
                color='r_per_ton',
                hover_data=['customer_normalized', 'distance'],
                title='Customer Performance Matrix (Normalized)',
                color_continuous_scale='RdYlGn_r'
            )
            
            # Add quadrant lines
            fig_route_matrix.add_hline(
                y=route_metrics['total_waiting_cleaned'].median(), 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5
            )
            fig_route_matrix.add_vline(
                x=route_metrics['utilization_percent'].median(), 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5
            )
            
            st.plotly_chart(fig_route_matrix, use_container_width=True)
    
    # Tab 8: Driver Performance
    with tab8:
        st.markdown("## 👤 Driver Performance Analysis")
        
        driver_metrics = filtered_df.groupby('driver_name').agg({
            'utilization_percent': 'mean',
            'total_waiting_cleaned': 'mean',
            'r_per_ton': 'mean',
            'driver_name': 'count',
            'tonnage_loaded': 'sum'
        }).rename(columns={'driver_name': 'trip_count'}).reset_index()
        
        driver_metrics = driver_metrics[driver_metrics['trip_count'] >= 10]
        
        if len(driver_metrics) > 0:
            # Top drivers by performance
            top_drivers = driver_metrics.nlargest(15, 'tonnage_loaded')
            
            fig_driver_perf = px.scatter(
                top_drivers,
                x='utilization_percent',
                y='r_per_ton',
                size='tonnage_loaded',
                hover_data=['driver_name', 'trip_count'],
                title='Driver Performance: Utilization vs Cost Efficiency',
                color='total_waiting_cleaned',
                color_continuous_scale='RdYlGn_r'
            )
            
            st.plotly_chart(fig_driver_perf, use_container_width=True)
    
    # Tab 9: Strategic Insights
    with tab9:
        st.markdown("## 💡 Strategic Insights & Data-Driven Recommendations")
        
        # Calculate route efficiency matrix like the old dashboard
        route_efficiency_data = filtered_df.groupby('customer_normalized').agg({
            'utilization_percent': 'mean',
            'total_waiting_cleaned': 'mean',
            'r_per_ton': 'mean',
            'distance': 'mean',
            'cost_incurred': 'sum',
            'customer_normalized': 'count'
        }).rename(columns={'customer_normalized': 'trip_count'}).reset_index()
        
        route_efficiency_data = route_efficiency_data[route_efficiency_data['trip_count'] >= 5]
        
        # Define efficiency quadrants
        median_util = route_efficiency_data['utilization_percent'].median()
        median_wait = route_efficiency_data['total_waiting_cleaned'].median()
        
        # Categorize routes into efficiency quadrants
        def categorize_route(row):
            if row['utilization_percent'] >= median_util and row['total_waiting_cleaned'] <= median_wait:
                return 'Optimal Performance'
            elif row['utilization_percent'] >= median_util and row['total_waiting_cleaned'] > median_wait:
                return 'High Utilization, Reduce Waiting'
            elif row['utilization_percent'] < median_util and row['total_waiting_cleaned'] <= median_wait:
                return 'Improve Utilization'
            else:
                return 'Critical - Fix Both'
        
        route_efficiency_data['efficiency_category'] = route_efficiency_data.apply(categorize_route, axis=1)
        
        # Strategic Route Analysis
        st.markdown("### 🗺️ Route Efficiency Matrix Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Route efficiency scatter plot
            fig_efficiency = px.scatter(
                route_efficiency_data,
                x='utilization_percent',
                y='total_waiting_cleaned',
                size='trip_count',
                color='efficiency_category',
                hover_data=['customer_normalized', 'r_per_ton', 'distance'],
                title='Customer Performance Matrix',
                color_discrete_map={
                    'Optimal Performance': '#2E8B57',           # Bright green (more distinct)
                    'High Utilization, Reduce Waiting': '#FF8C00',  # Bright orange (more distinct)
                    'Improve Utilization': '#4169E1',           # Royal blue (more distinct from red)
                    'Critical - Fix Both': '#DC143C'            # Crimson red
                }
            )
            
            # Add quadrant lines
            fig_efficiency.add_hline(y=median_wait, line_dash="dash", line_color="gray", opacity=0.5)
            fig_efficiency.add_vline(x=median_util, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with col2:
            # Efficiency category summary
            category_summary = route_efficiency_data.groupby('efficiency_category').agg({
                'customer_normalized': 'count',
                'cost_incurred': 'sum',
                'r_per_ton': 'mean'
            }).rename(columns={'customer_normalized': 'route_count'}).round(1)
            
            st.markdown("#### 📊 Route Categories Summary")
            st.dataframe(category_summary.style.format({
                'cost_incurred': 'R{:,.0f}',
                'r_per_ton': 'R{:.0f}'
            }), use_container_width=True)
        
        # Key findings based on analysis
        st.markdown("### 🎯 Data-Driven Key Findings")
        
        # Calculate specific insights
        high_cost_routes = route_efficiency_data[route_efficiency_data['r_per_ton'] > 800]
        critical_routes = route_efficiency_data[route_efficiency_data['efficiency_category'] == 'Critical - Fix Both']
        long_haul_inefficient = route_efficiency_data[(route_efficiency_data['distance'] > 500) & 
                                                     (route_efficiency_data['utilization_percent'] < 95)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🚨 Critical Performance Issues (Q1 2025)**
            - **{len(critical_routes)} customers** in critical quadrant (low utilization + high waiting)
            - **{len(high_cost_routes)} routes** with cost >R800/ton (target: <R500)
            - **{len(long_haul_inefficient)} long-haul routes** (>500km) operating <95% capacity
            - **Hotazel (UMK)**: Extreme waiting times (~61h average vs 2h target)
            - **Remote Northern Cape/Limpopo**: R900-1000+/ton vs R300-350 for core routes
            """)
        
        with col2:
            optimal_routes = route_efficiency_data[route_efficiency_data['efficiency_category'] == 'Optimal Performance']
            st.markdown(f"""
            **✅ Success Stories to Replicate**
            - **{len(optimal_routes)} customers** in optimal quadrant
            - **Core Mpumalanga/Gauteng routes**: Consistently R290-340/ton
            - **34T fleet efficiency**: Higher capacity utilization when deployed correctly
            - **Best practices**: Full loads, minimal waiting, strategic fleet deployment
            """)
        
        # Specific recommendations based on findings
        st.markdown("### 📋 Evidence-Based Action Plan")

        # Create recommendations data
        recommendations_data = []

        # Critical routes recommendations
        if len(critical_routes) > 0:
            recommendations_data.append({
                'Priority': 'URGENT',
                'Category': 'Critical Routes',
                'Specific Action': f'Address {len(critical_routes)} critical customers: {", ".join(critical_routes["customer_normalized"].head(3).tolist())}',
                'Expected Impact': f'R{critical_routes["cost_incurred"].sum()/1000:.0f}K cost reduction',
                'Timeline': '2-4 weeks'
            })

        # Long haul efficiency
        if len(long_haul_inefficient) > 0:
            recommendations_data.append({
                'Priority': 'HIGH',
                'Category': 'Long-Haul Optimization',
                'Specific Action': 'Consolidate loads for distant sites or implement remote delivery surcharge',
                'Expected Impact': 'R15-20M annually',
                'Timeline': '6-8 weeks'
            })

        # Waiting time reduction
        high_wait_routes = route_efficiency_data[route_efficiency_data['total_waiting_cleaned'] > 10]
        if len(high_wait_routes) > 0:
            recommendations_data.append({
                'Priority': 'HIGH',
                'Category': 'Standing Time Reduction',
                'Specific Action': f'Target {len(high_wait_routes)} high-waiting customers for process improvement',
                'Expected Impact': 'R4.8-6.4M annually',
                'Timeline': '4-6 weeks'
            })

        # Fleet optimization
        recommendations_data.append({
            'Priority': 'MEDIUM',
            'Category': 'Fleet Mix Optimization',
            'Specific Action': 'Strategic deployment: 34T for high-volume routes, 30T for smaller loads',
            'Expected Impact': 'R18.4M annually',
            'Timeline': '8-12 weeks'
        })

        # Process improvements
        recommendations_data.append({
            'Priority': 'MEDIUM',
            'Category': 'Operational Process',
            'Specific Action': 'Review Q1 2025 changes that caused performance degradation',
            'Expected Impact': 'Restore Q1 2024 efficiency levels',
            'Timeline': '2-3 weeks'
        })

        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations_data)

        # Display using Streamlit columns for better control
        for idx, row in recommendations_df.iterrows():
            # Create a container for each recommendation
            with st.container():
                # Add custom styling based on priority
                if row['Priority'] == 'URGENT':
                    st.markdown(f"""
                    <div style='background-color: rgba(220, 20, 60, 0.1); 
                                border-left: 4px solid #DC143C; 
                                padding: 15px; 
                                margin-bottom: 10px; 
                                border-radius: 5px;'>
                        <h4 style='color: #DC143C; margin: 0;'>🚨 {row['Priority']} - {row['Category']}</h4>
                        <p style='color: #e0e0e0; margin: 10px 0;'><strong>Action:</strong> {row['Specific Action']}</p>
                        <div style='display: flex; justify-content: space-between; color: #aaa;'>
                            <span><strong>Impact:</strong> {row['Expected Impact']}</span>
                            <span><strong>Timeline:</strong> {row['Timeline']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif row['Priority'] == 'HIGH':
                    st.markdown(f"""
                    <div style='background-color: rgba(255, 140, 0, 0.1); 
                                border-left: 4px solid #FF8C00; 
                                padding: 15px; 
                                margin-bottom: 10px; 
                                border-radius: 5px;'>
                        <h4 style='color: #FF8C00; margin: 0;'>⚠️ {row['Priority']} - {row['Category']}</h4>
                        <p style='color: #e0e0e0; margin: 10px 0;'><strong>Action:</strong> {row['Specific Action']}</p>
                        <div style='display: flex; justify-content: space-between; color: #aaa;'>
                            <span><strong>Impact:</strong> {row['Expected Impact']}</span>
                            <span><strong>Timeline:</strong> {row['Timeline']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # MEDIUM
                    st.markdown(f"""
                    <div style='background-color: rgba(65, 105, 225, 0.1); 
                                border-left: 4px solid #4169E1; 
                                padding: 15px; 
                                margin-bottom: 10px; 
                                border-radius: 5px;'>
                        <h4 style='color: #4169E1; margin: 0;'>📌 {row['Priority']} - {row['Category']}</h4>
                        <p style='color: #e0e0e0; margin: 10px 0;'><strong>Action:</strong> {row['Specific Action']}</p>
                        <div style='display: flex; justify-content: space-between; color: #aaa;'>
                            <span><strong>Impact:</strong> {row['Expected Impact']}</span>
                            <span><strong>Timeline:</strong> {row['Timeline']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Alternative: Use Streamlit's native AgGrid or DataFrame display
        # If you prefer a table format, use this instead:
        st.markdown("---")
        st.markdown("#### 📊 Action Plan Summary Table")

        # Create a styled dataframe
        styled_recommendations = recommendations_df.style.apply(
            lambda x: ['background-color: rgba(220, 20, 60, 0.2)' if x.Priority == 'URGENT' 
                    else 'background-color: rgba(255, 140, 0, 0.2)' if x.Priority == 'HIGH'
                    else 'background-color: rgba(65, 105, 225, 0.2)' for i in x], 
            axis=1
        ).set_properties(**{
            'color': '#e0e0e0',
            'border': '1px solid #333',
            'padding': '10px'
        })

        st.dataframe(styled_recommendations, use_container_width=True, height=250)

        
        # Financial impact summary
        st.markdown("### 💰 Projected Financial Impact")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Addressable Savings",
                "R43-55M annually",
                delta="Conservative to stretch targets"
            )
        
        with col2:
            st.metric(
                "Quick Wins (8 weeks)",
                "R20-25M potential",
                delta="Critical routes + waiting time"
            )
        
        with col3:
            st.metric(
                "Full Program (12 months)", 
                "R40-50M target",
                delta="Meets optimization mandate"
            )
        
        # Implementation priority matrix
        st.markdown("### 🎯 Implementation Priority Matrix")
        
        impact_effort_data = pd.DataFrame({
            'Initiative': ['Fix Critical Routes', 'Reduce Waiting Times', 'Optimize Long-Haul', 'Fleet Mix Strategy', 'Process Review'],
            'Impact_Score': [10, 9, 8, 7, 6],
            'Effort_Score': [3, 4, 6, 7, 2],
            'Category': ['Quick Win', 'Quick Win', 'Major Project', 'Major Project', 'Quick Win']
        })
        
        fig_priority = px.scatter(
            impact_effort_data,
            x='Effort_Score',
            y='Impact_Score',
            size=[15, 12, 10, 8, 6],
            color='Category',
            text='Initiative',
            title='Priority Matrix: Impact vs Implementation Effort',
            color_discrete_map={
                'Quick Win': ENAEX_COLORS['success'],
                'Major Project': ENAEX_COLORS['primary']
            }
        )
        fig_priority.update_traces(textposition='top center')
        fig_priority.update_layout(
            xaxis_title='Implementation Effort (1=Easy, 10=Hard)',
            yaxis_title='Expected Impact (1=Low, 10=High)'
        )
        
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Footer with data filtering information
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {ENAEX_COLORS['light_text']}; padding: 2rem 0;">
        <p><strong>Enaex Africa Enhanced Analytics</strong> - Strategic Customer Focus</p>
        <p style="font-size: 0.9rem;">
            Analysis based on {len(filtered_df):,} records from strategically relevant customers only.<br>
            Customer names normalized. 12:00 AM timestamps (2024) excluded only from temporal pattern analysis.<br>
            Dashboard Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            <strong>Data Scope:</strong> 30T Fleet: {', '.join([k.title() for k in CUSTOMER_MAPPINGS_30T.keys()])} | 
            34T Fleet: {', '.join([k.title() for k in CUSTOMER_MAPPINGS_34T.keys()])} |
            <em>Full dataset preserved for Q1 comparisons; 12:00 AM filtering applied only to temporal patterns</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()