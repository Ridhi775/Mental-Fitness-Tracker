import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Mental Fitness Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Red & Black Theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #8B0000 0%, #1a0000 50%, #000000 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(139, 0, 0, 0.5);
        border: 1px solid #8B0000;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        color: #ff1a1a;
        text-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: #cccccc;
    }
    .badge {
        background: rgba(139, 0, 0, 0.6);
        color: #ff4444;
        padding: 0.2rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid #8B0000;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a0000 0%, #2a0000 100%);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(139, 0, 0, 0.3);
        border-left: 4px solid #8B0000;
        margin-bottom: 0.5rem;
        border: 1px solid #3a0000;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(139, 0, 0, 0.5);
        transition: all 0.3s ease;
        border-color: #ff1a1a;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ff1a1a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #999999;
        font-weight: 500;
    }
    .success-box {
        background: linear-gradient(135deg, #0a1a0a 0%, #1a2a1a 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff00;
        margin: 1rem 0;
        color: #00ff00;
        border: 1px solid #003300;
    }
    .info-box {
        background: linear-gradient(135deg, #000a1a 0%, #001a2a 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0066ff;
        margin: 1rem 0;
        color: #66aaff;
        border: 1px solid #003366;
    }
    .footer {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0000 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        border: 1px solid #1a0000;
        color: #666666;
    }
    .footer strong {
        color: #ff1a1a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #0a0a0a;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border: 1px solid #1a0000;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #666666;
        background: #0a0a0a;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B0000 0%, #1a0000 100%);
        color: #ff4444;
        border: 1px solid #8B0000;
    }
    .stButton > button {
        background: linear-gradient(135deg, #8B0000 0%, #1a0000 100%);
        color: #ff4444;
        border: 1px solid #8B0000;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 0, 0, 0.5);
        border-color: #ff1a1a;
        color: #ff6666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Mental Fitness Tracker</h1>
    <p>Comprehensive Mental Health Analytics & Visualization Platform</p>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap;">
        <span class="badge">📊 Overview</span>
        <span class="badge">📈 Trends</span>
        <span class="badge">🤖 AI</span>
        <span class="badge">🌍 Global</span>
        <span class="badge">📊 Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a0000 0%, #2a0000 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center;
                border: 1px solid #8B0000;">
        <h3 style="margin: 0; color: #ff1a1a;">📁 Data Center</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload your mental health dataset"
    )
    
    if st.button("📥 Load Sample Dataset", use_container_width=True):
        try:
            sample_paths = [
                "data/mental-and-substance-use-as-share-of-disease.csv",
                "mental-and-substance-use-as-share-of-disease.csv",
                "../data/mental-and-substance-use-as-share-of-disease.csv"
            ]
            
            sample_loaded = False
            for path in sample_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.session_state.df = df
                    st.session_state.uploaded_file_name = os.path.basename(path)
                    st.session_state.data_loaded = True
                    sample_loaded = True
                    st.success("✅ Sample data loaded successfully!")
                    st.rerun()
                    break
            
            if not sample_loaded:
                st.error("❌ Sample data not found. Please upload your own CSV.")
        except Exception as e:
            st.error(f"❌ Error loading sample: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📊 Data Status")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"✅ Loaded: {st.session_state.uploaded_file_name}")
        st.info(f"📊 {len(df):,} rows • {len(df.columns)} columns")
        
        with st.expander("📋 View Columns"):
            for col in df.columns:
                st.write(f"- {col}")
        
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.uploaded_file_name = None
            st.session_state.data_loaded = False
            st.session_state.processed_data = None
            st.rerun()
    else:
        st.warning("⚠️ No data loaded")
        st.info("Upload a CSV or load sample data to begin")

# Main Content
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = df.loc[:, ~df.columns.duplicated()]
        st.session_state.df = df
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.data_loaded = True
        
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>File uploaded successfully!</strong><br>
            📄 {uploaded_file.name} • 📊 {len(df):,} rows • 📋 {len(df.columns)} columns
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Process and Display Data
if st.session_state.df is not None:
    df = st.session_state.df
    df = df.loc[:, ~df.columns.duplicated()]
    
    @st.cache_data
    def process_data(df):
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip()
        
        entity_col = None
        year_col = None
        dalys_col = None
        
        for col in df_clean.columns:
            col_lower = col.lower()
            if 'entity' in col_lower or 'country' in col_lower or 'location' in col_lower or 'region' in col_lower:
                entity_col = col
            elif 'year' in col_lower or 'yr' in col_lower:
                year_col = col
            elif 'dalys' in col_lower or 'burden' in col_lower or 'disability' in col_lower or 'value' in col_lower:
                dalys_col = col
        
        if entity_col is None and len(df_clean.columns) >= 1:
            entity_col = df_clean.columns[0]
        if year_col is None and len(df_clean.columns) >= 2:
            year_col = df_clean.columns[1]
        if dalys_col is None and len(df_clean.columns) >= 3:
            dalys_col = df_clean.columns[2]
        
        rename_map = {}
        if entity_col:
            rename_map[entity_col] = 'Entity'
        if year_col:
            rename_map[year_col] = 'Year'
        if dalys_col:
            rename_map[dalys_col] = 'DALYs'
        
        if rename_map:
            df_clean = df_clean.rename(columns=rename_map)
        
        if 'Year' in df_clean.columns:
            df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        if 'DALYs' in df_clean.columns:
            df_clean['DALYs'] = pd.to_numeric(df_clean['DALYs'], errors='coerce')
        
        if 'Entity' in df_clean.columns and 'Year' in df_clean.columns and 'DALYs' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['Entity', 'Year', 'DALYs'])
        
        return df_clean
    
    try:
        df_clean = process_data(df)
        st.session_state.processed_data = df_clean
        
        has_entity = 'Entity' in df_clean.columns
        has_year = 'Year' in df_clean.columns
        has_dalys = 'DALYs' in df_clean.columns
        
        # Show data preview
        st.markdown("### 📄 Data Preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Quick Stats
        st.markdown("### 📊 Quick Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 Total Records</div>
                <div class="metric-value">{len(df_clean):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            entities = df_clean['Entity'].nunique() if has_entity else 'N/A'
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #cc0000;">
                <div class="metric-label">🌍 Unique Entities</div>
                <div class="metric-value">{entities}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if has_dalys and len(df_clean['DALYs']) > 0:
                avg_dalys = df_clean['DALYs'].mean()
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ff1a1a;">
                    <div class="metric-label">📈 Average DALYs</div>
                    <div class="metric-value">{avg_dalys:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ff1a1a;">
                    <div class="metric-label">📈 Average DALYs</div>
                    <div class="metric-value">N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if has_year and len(df_clean['Year']) > 0:
                years = f"{int(df_clean['Year'].min())} - {int(df_clean['Year'].max())}"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #990000;">
                    <div class="metric-label">📅 Time Period</div>
                    <div class="metric-value">{years}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #990000;">
                    <div class="metric-label">📅 Time Period</div>
                    <div class="metric-value">N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.info(f"Detected columns: Entity: {has_entity}, Year: {has_year}, DALYs: {has_dalys}")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "📈 Trends",
            "📉 Analysis",
            "🤖 AI Studio",
            "🌍 Global",
            "📊 Correlation",
            "📋 Data Explorer"
        ])
        
        # ============ TAB 1: OVERVIEW ============
        with tab1:
            st.markdown("### 📊 Overview Dashboard")
            
            if has_entity and has_year and has_dalys:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    entities = sorted(df_clean['Entity'].unique())
                    selected_entity = st.selectbox("Select Entity", entities, key="overview_entity")
                
                with col2:
                    year_min = int(df_clean['Year'].min())
                    year_max = int(df_clean['Year'].max())
                    year_range = st.slider(
                        "Year Range",
                        year_min, year_max,
                        (year_min, year_max),
                        key="overview_year"
                    )
                
                with col3:
                    dalys_min = float(df_clean['DALYs'].min())
                    dalys_max = float(df_clean['DALYs'].max())
                    dalys_range = st.slider(
                        "DALYs Range",
                        dalys_min, dalys_max,
                        (dalys_min, dalys_max),
                        key="overview_dalys"
                    )
                
                filtered_df = df_clean[
                    (df_clean['Entity'] == selected_entity) &
                    (df_clean['Year'] >= year_range[0]) &
                    (df_clean['Year'] <= year_range[1]) &
                    (df_clean['DALYs'] >= dalys_range[0]) &
                    (df_clean['DALYs'] <= dalys_range[1])
                ]
                
                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values('Year')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean DALYs", f"{filtered_df['DALYs'].mean():.2f}")
                    with col2:
                        st.metric("Max DALYs", f"{filtered_df['DALYs'].max():.2f}")
                    with col3:
                        st.metric("Min DALYs", f"{filtered_df['DALYs'].min():.2f}")
                    with col4:
                        st.metric("Data Points", len(filtered_df))
                    
                    st.markdown("#### 📈 DALYs Trend Over Time")
                    
                    # Using matplotlib instead of plotly to avoid fetch errors
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0a0a0a')
                    
                    ax.plot(filtered_df['Year'], filtered_df['DALYs'], 
                           color='#ff1a1a', linewidth=2, marker='o', 
                           markersize=8, markerfacecolor='#8B0000',
                           markeredgecolor='#ff1a1a', markeredgewidth=2)
                    ax.fill_between(filtered_df['Year'], filtered_df['DALYs'], 
                                   alpha=0.3, color='#8B0000')
                    ax.set_xlabel('Year', color='#666666')
                    ax.set_ylabel('DALYs', color='#666666')
                    ax.set_title(f'Mental Health Burden - {selected_entity}', color='#ff1a1a')
                    ax.tick_params(colors='#666666')
                    ax.grid(True, alpha=0.2, color='#1a0000')
                    
                    st.pyplot(fig)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Distribution")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        ax.hist(filtered_df['DALYs'], bins=30, color='#8B0000', 
                               edgecolor='#ff1a1a', alpha=0.7)
                        ax.set_xlabel('DALYs', color='#666666')
                        ax.set_ylabel('Frequency', color='#666666')
                        ax.set_title('DALYs Distribution', color='#ff1a1a')
                        ax.tick_params(colors='#666666')
                        ax.grid(True, alpha=0.2, color='#1a0000')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("#### 📈 Year-over-Year Change")
                        if len(filtered_df) > 1:
                            filtered_df['Change'] = filtered_df['DALYs'].pct_change() * 100
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            fig.patch.set_facecolor('#0a0a0a')
                            ax.set_facecolor('#0a0a0a')
                            
                            colors = ['#8B0000' if x < 0 else '#ff1a1a' for x in filtered_df['Change'].fillna(0)]
                            ax.bar(filtered_df['Year'], filtered_df['Change'].fillna(0), 
                                  color=colors, edgecolor='#ff1a1a', linewidth=1)
                            ax.axhline(y=0, color='#666666', linestyle='--', linewidth=1)
                            ax.set_xlabel('Year', color='#666666')
                            ax.set_ylabel('Change (%)', color='#666666')
                            ax.set_title('Year-over-Year Change (%)', color='#ff1a1a')
                            ax.tick_params(colors='#666666')
                            ax.grid(True, alpha=0.2, color='#1a0000')
                            
                            st.pyplot(fig)
                else:
                    st.warning("No data available for the selected filters")
            else:
                st.warning("⚠️ Missing required columns. Please ensure your data has Entity, Year, and DALYs columns.")
        
        # ============ TAB 2: TRENDS ============
        with tab2:
            st.markdown("### 📈 Trend Analysis")
            
            if has_entity and has_year and has_dalys:
                entities_multi = st.multiselect(
                    "Select entities to compare",
                    sorted(df_clean['Entity'].unique()),
                    default=sorted(df_clean['Entity'].unique())[:3] if len(df_clean['Entity'].unique()) >= 3 else [],
                    key="trend_entities"
                )
                
                if entities_multi:
                    compare_df = df_clean[df_clean['Entity'].isin(entities_multi)]
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0a0a0a')
                    
                    colors = ['#ff1a1a', '#cc0000', '#990000', '#660000', '#330000']
                    for i, entity in enumerate(entities_multi):
                        entity_data = compare_df[compare_df['Entity'] == entity].sort_values('Year')
                        ax.plot(entity_data['Year'], entity_data['DALYs'], 
                               color=colors[i % len(colors)], linewidth=2, 
                               marker='o', label=entity, markersize=6)
                    
                    ax.set_xlabel('Year', color='#666666')
                    ax.set_ylabel('DALYs', color='#666666')
                    ax.set_title('Mental Health Burden Comparison', color='#ff1a1a')
                    ax.tick_params(colors='#666666')
                    ax.grid(True, alpha=0.2, color='#1a0000')
                    ax.legend(loc='best', facecolor='#0a0a0a', edgecolor='#1a0000', labelcolor='#666666')
                    
                    st.pyplot(fig)
                    
                    st.markdown("#### 📊 Summary Statistics")
                    summary = compare_df.groupby('Entity')['DALYs'].agg(['mean', 'min', 'max', 'std']).round(2)
                    summary.columns = ['Mean', 'Min', 'Max', 'Std Dev']
                    st.dataframe(summary, use_container_width=True)
                    
                    st.markdown("#### 📊 Distribution Comparison")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0a0a0a')
                    
                    data_to_plot = [compare_df[compare_df['Entity'] == entity]['DALYs'] for entity in entities_multi]
                    bp = ax.boxplot(data_to_plot, labels=entities_multi, patch_artist=True)
                    
                    for patch, color in zip(bp['boxes'], colors[:len(entities_multi)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_xlabel('Entity', color='#666666')
                    ax.set_ylabel('DALYs', color='#666666')
                    ax.set_title('DALYs Distribution by Entity', color='#ff1a1a')
                    ax.tick_params(colors='#666666')
                    ax.grid(True, alpha=0.2, color='#1a0000')
                    
                    st.pyplot(fig)
                else:
                    st.info("Select at least one entity to compare")
            else:
                st.warning("⚠️ Missing required columns. Please ensure your data has Entity, Year, and DALYs columns.")
        
        # ============ TAB 3: ANALYSIS ============
        with tab3:
            st.markdown("### 📉 Advanced Analysis")
            
            if has_dalys:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Statistical Summary")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 'Count'],
                        'Value': [
                            df_clean['DALYs'].mean(),
                            df_clean['DALYs'].median(),
                            df_clean['DALYs'].std(),
                            df_clean['DALYs'].min(),
                            df_clean['DALYs'].max(),
                            df_clean['DALYs'].max() - df_clean['DALYs'].min(),
                            len(df_clean)
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Box Plot Analysis")
                    if has_entity:
                        top_entities = df_clean.groupby('Entity')['DALYs'].mean().nlargest(10).index
                        box_df = df_clean[df_clean['Entity'].isin(top_entities)]
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        data_to_plot = [box_df[box_df['Entity'] == entity]['DALYs'] for entity in top_entities]
                        colors = ['#ff1a1a', '#cc0000', '#990000', '#660000', '#330000'] * 2
                        bp = ax.boxplot(data_to_plot, labels=top_entities, patch_artist=True)
                        
                        for patch, color in zip(bp['boxes'], colors[:len(top_entities)]):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.set_xlabel('Entity', color='#666666')
                        ax.set_ylabel('DALYs', color='#666666')
                        ax.set_title('Top 10 Entities Distribution', color='#ff1a1a')
                        ax.tick_params(colors='#666666')
                        ax.grid(True, alpha=0.2, color='#1a0000')
                        plt.xticks(rotation=45, ha='right')
                        
                        st.pyplot(fig)
                
                if has_entity:
                    st.markdown("#### 🎯 Cluster Analysis")
                    cluster_df = df_clean.groupby('Entity')['DALYs'].mean().reset_index()
                    
                    if len(cluster_df) > 5:
                        kmeans = KMeans(n_clusters=min(4, len(cluster_df)//2), random_state=42)
                        clusters = kmeans.fit_predict(cluster_df[['DALYs']])
                        cluster_df['Cluster'] = clusters
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        scatter = ax.scatter(cluster_df['Entity'], cluster_df['DALYs'], 
                                           c=cluster_df['Cluster'], cmap='Reds', 
                                           s=cluster_df['DALYs']*50, alpha=0.7,
                                           edgecolors='#ff1a1a', linewidth=1)
                        ax.set_xlabel('Entity', color='#666666')
                        ax.set_ylabel('DALYs', color='#666666')
                        ax.set_title('Entity Clustering by DALYs', color='#ff1a1a')
                        ax.tick_params(colors='#666666')
                        plt.xticks(rotation=45, ha='right')
                        
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Missing 'DALYs' column for analysis.")
        
        # ============ TAB 4: AI STUDIO ============
        with tab4:
            st.markdown("### 🤖 AI Studio - Machine Learning")
            
            if has_dalys and len(df_clean) > 10:
                try:
                    ml_df = df_clean.copy()
                    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 1:
                        le = LabelEncoder()
                        if 'Entity' in ml_df.columns:
                            ml_df['Entity_encoded'] = le.fit_transform(ml_df['Entity'].astype(str))
                        
                        feature_cols = []
                        if 'Entity_encoded' in ml_df.columns:
                            feature_cols.append('Entity_encoded')
                        if 'Year' in ml_df.columns:
                            feature_cols.append('Year')
                        
                        for col in numeric_cols:
                            if col not in ['DALYs'] and col not in feature_cols:
                                feature_cols.append(col)
                        
                        if len(feature_cols) >= 1:
                            X = ml_df[feature_cols]
                            y = ml_df['DALYs']
                            
                            X = X.fillna(0)
                            y = y.fillna(0)
                            
                            with st.spinner("Training model..."):
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42
                                )
                                
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("R² Score", f"{r2:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                with col3:
                                    st.metric("MAE", f"{mae:.4f}")
                                with col4:
                                    st.metric("MSE", f"{mse:.4f}")
                                
                                st.markdown("#### 📊 Actual vs Predicted")
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                fig.patch.set_facecolor('#0a0a0a')
                                ax.set_facecolor('#0a0a0a')
                                
                                ax.scatter(y_test, y_pred, alpha=0.6, color='#8B0000', 
                                         edgecolors='#ff1a1a', linewidth=1)
                                ax.plot([y_test.min(), y_test.max()], 
                                       [y_test.min(), y_test.max()], 
                                       'r--', linewidth=2, label='Perfect Prediction')
                                ax.set_xlabel('Actual', color='#666666')
                                ax.set_ylabel('Predicted', color='#666666')
                                ax.set_title('Actual vs Predicted', color='#ff1a1a')
                                ax.tick_params(colors='#666666')
                                ax.grid(True, alpha=0.2, color='#1a0000')
                                ax.legend(loc='best', facecolor='#0a0a0a', edgecolor='#1a0000', labelcolor='#666666')
                                
                                st.pyplot(fig)
                                
                                if hasattr(model, 'feature_importances_'):
                                    st.markdown("#### 🔑 Feature Importance")
                                    importance_df = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=True)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    fig.patch.set_facecolor('#0a0a0a')
                                    ax.set_facecolor('#0a0a0a')
                                    
                                    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
                                    for bar in bars:
                                        bar.set_color('#8B0000')
                                        bar.set_edgecolor('#ff1a1a')
                                        bar.set_linewidth(1)
                                    
                                    ax.set_xlabel('Importance', color='#666666')
                                    ax.set_title('Feature Importance', color='#ff1a1a')
                                    ax.tick_params(colors='#666666')
                                    ax.grid(True, alpha=0.2, color='#1a0000')
                                    
                                    st.pyplot(fig)
                                
                                st.success("✅ Model trained successfully!")
                        else:
                            st.warning("Not enough features for ML. Need at least 1 feature column.")
                    else:
                        st.warning("Need at least 2 numeric columns for ML")
                except Exception as e:
                    st.error(f"⚠️ ML Error: {str(e)}")
                    st.info("Try using the sample dataset or ensure your data has numeric columns.")
            else:
                st.warning("Need at least 10 rows of data with 'DALYs' column")
        
        # ============ TAB 5: GLOBAL ============
        with tab5:
            st.markdown("### 🌍 Global View")
            
            if has_entity and has_dalys:
                try:
                    if has_year:
                        selected_year = st.selectbox(
                            "Select Year",
                            sorted(df_clean['Year'].unique(), reverse=True),
                            key="global_year"
                        )
                        global_df = df_clean[df_clean['Year'] == selected_year]
                    else:
                        global_df = df_clean
                    
                    if not global_df.empty:
                        # Use matplotlib for world map (simpler, no external dependencies)
                        st.markdown("#### 🌍 World Map View")
                        
                        # Create a pivot table for the map
                        top_countries = global_df.nlargest(20, 'DALYs')[['Entity', 'DALYs']]
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        # Create bar chart instead of map (more reliable)
                        colors_red = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_countries)))
                        bars = ax.barh(top_countries['Entity'], top_countries['DALYs'], color=colors_red)
                        ax.set_xlabel('DALYs', color='#666666')
                        ax.set_title(f'Global Mental Health Burden' + (f' ({selected_year})' if has_year else ''), color='#ff1a1a')
                        ax.tick_params(colors='#666666')
                        ax.grid(True, alpha=0.2, color='#1a0000')
                        
                        st.pyplot(fig)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🏆 Top 10 Countries")
                            top_countries = global_df.nlargest(10, 'DALYs')[['Entity', 'DALYs']]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            fig.patch.set_facecolor('#0a0a0a')
                            ax.set_facecolor('#0a0a0a')
                            
                            colors_red = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_countries)))
                            bars = ax.barh(top_countries['Entity'], top_countries['DALYs'], color=colors_red)
                            ax.set_xlabel('DALYs', color='#666666')
                            ax.set_title('Highest Mental Health Burden', color='#ff1a1a')
                            ax.tick_params(colors='#666666')
                            ax.grid(True, alpha=0.2, color='#1a0000')
                            
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### 📉 Bottom 10 Countries")
                            bottom_countries = global_df.nsmallest(10, 'DALYs')[['Entity', 'DALYs']]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            fig.patch.set_facecolor('#0a0a0a')
                            ax.set_facecolor('#0a0a0a')
                            
                            colors_green = plt.cm.Greens(np.linspace(0.3, 0.8, len(bottom_countries)))
                            bars = ax.barh(bottom_countries['Entity'], bottom_countries['DALYs'], color=colors_green)
                            ax.set_xlabel('DALYs', color='#666666')
                            ax.set_title('Lowest Mental Health Burden', color='#ff1a1a')
                            ax.tick_params(colors='#666666')
                            ax.grid(True, alpha=0.2, color='#1a0000')
                            
                            st.pyplot(fig)
                        
                        # Global Statistics
                        st.markdown("#### 📊 Global Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Global Mean", f"{global_df['DALYs'].mean():.2f}")
                        with col2:
                            st.metric("Global Median", f"{global_df['DALYs'].median():.2f}")
                        with col3:
                            st.metric("Global Max", f"{global_df['DALYs'].max():.2f}")
                        with col4:
                            st.metric("Global Min", f"{global_df['DALYs'].min():.2f}")
                    else:
                        st.warning("No data available for the selected year")
                except Exception as e:
                    st.error(f"⚠️ Global view error: {str(e)}")
                    st.info("Make sure your Entity column contains valid country names.")
            else:
                st.warning("⚠️ Global view requires 'Entity' and 'DALYs' columns.")
        
        # ============ TAB 6: CORRELATION ============
        with tab6:
            st.markdown("### 📊 Correlation Analysis")
            
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    fig.patch.set_facecolor('#0a0a0a')
                    ax.set_facecolor('#0a0a0a')
                    
                    corr_matrix = df_clean[numeric_cols].corr()
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap='RdBu_r',
                        center=0,
                        square=True,
                        linewidths=2,
                        ax=ax,
                        annot_kws={'color': 'white', 'size': 10}
                    )
                    ax.set_title('Correlation Matrix', color='#ff1a1a', size=14)
                    ax.tick_params(colors='#666666')
                    
                    st.pyplot(fig)
                    
                    st.markdown("#### 🔍 Top Correlations")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(corr_df.head(10), use_container_width=True)
                    
                    st.markdown("#### 📈 Scatter Plot")
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("Select Variable 1", numeric_cols, key="corr_var1")
                    with col2:
                        var2 = st.selectbox("Select Variable 2", numeric_cols, key="corr_var2")
                    
                    if var1 and var2 and var1 != var2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_facecolor('#0a0a0a')
                        ax.set_facecolor('#0a0a0a')
                        
                        ax.scatter(df_clean[var1], df_clean[var2], alpha=0.6, 
                                  c='#8B0000', edgecolors='#ff1a1a', linewidth=0.5)
                        ax.set_xlabel(var1, color='#666666')
                        ax.set_ylabel(var2, color='#666666')
                        ax.set_title(f'{var1} vs {var2}', color='#ff1a1a')
                        ax.tick_params(colors='#666666')
                        
                        corr_val = df_clean[var1].corr(df_clean[var2])
                        ax.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', 
                               transform=ax.transAxes, color='#ff1a1a',
                               fontsize=12, verticalalignment='top')
                        
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"⚠️ Correlation error: {str(e)}")
            else:
                st.info(f"Need at least 2 numeric columns for correlation analysis. Found {len(numeric_cols)} numeric columns.")
                if len(numeric_cols) > 0:
                    st.write("Numeric columns found:", ", ".join(numeric_cols))
        
        # ============ TAB 7: DATA EXPLORER ============
        with tab7:
            st.markdown("### 📋 Data Explorer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                filter_col = st.selectbox("Select column to filter", df_clean.columns, key="explorer_col")
            
            with col2:
                if filter_col:
                    unique_vals = df_clean[filter_col].unique()
                    selected_vals = st.multiselect(
                        f"Select values",
                        unique_vals,
                        default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals,
                        key="explorer_vals"
                    )
            
            if filter_col and selected_vals:
                try:
                    filtered_explore = df_clean[df_clean[filter_col].isin(selected_vals)]
                    st.dataframe(filtered_explore, use_container_width=True, height=400)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = filtered_explore.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("📊 Stats", use_container_width=True):
                            st.dataframe(filtered_explore.describe(), use_container_width=True)
                    
                    with col3:
                        if st.button("📈 Visualize", use_container_width=True):
                            num_cols = filtered_explore.select_dtypes(include=[np.number]).columns
                            if len(num_cols) >= 2:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                fig.patch.set_facecolor('#0a0a0a')
                                ax.set_facecolor('#0a0a0a')
                                ax.scatter(
                                    filtered_explore[num_cols[0]],
                                    filtered_explore[num_cols[1]],
                                    alpha=0.6,
                                    c='#8B0000',
                                    edgecolors='#ff1a1a'
                                )
                                ax.set_xlabel(num_cols[0], color='#666666')
                                ax.set_ylabel(num_cols[1], color='#666666')
                                ax.set_title(f'{num_cols[0]} vs {num_cols[1]}', color='#ff1a1a')
                                ax.tick_params(colors='#666666')
                                st.pyplot(fig)
                            else:
                                st.warning("Need 2 numeric columns")
                except Exception as e:
                    st.error(f"⚠️ Explorer error: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("Please check your data format. The app automatically detects columns.")

# Footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span>🧠 <strong>Mental Fitness Tracker</strong> v2.0</span>
        <span>⚡ Powered by <strong>AI</strong></span>
        <span>📊 Real-time <strong>Analytics</strong></span>
        <span>🚀 Made with ❤️</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions for first-time users
if st.session_state.df is None:
    st.markdown("""
    <div class="info-box">
        <h4>🚀 Getting Started</h4>
        <ul>
            <li><strong>Upload CSV:</strong> Click "Browse files" to upload your mental health dataset</li>
            <li><strong>Sample Data:</strong> Click "Load Sample Dataset" to explore with example data</li>
            <li><strong>Analyze:</strong> Use the 7 tabs above to explore overview, trends, analysis, AI, global view, correlation, and data explorer</li>
            <li><strong>Export:</strong> Download filtered data and insights for your reports</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
