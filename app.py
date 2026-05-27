import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set Page Config
st.set_page_config(
    page_title="Palmer Penguins Analytics Dashboard",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Premium Design Aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Custom cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #eef2f6;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Research question boxes */
    .rq-box {
        background-color: #f7f9fc;
        border-left: 5px solid #1e3c72;
        border-radius: 4px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .rq-title {
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    .rq-explanation {
        font-size: 1rem;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def get_raw_data():
    return sns.load_dataset('penguins')

@st.cache_data
def get_cleaned_data(df_raw):
    df = df_raw.copy()
    # Drop rows where all measurements are null
    empty_rows = df[df['bill_length_mm'].isna() & df['body_mass_g'].isna()]
    df = df.drop(empty_rows.index).reset_index(drop=True)
    
    # Impute missing sex based on species median
    missing_sex_mask = df['sex'].isna()
    for idx in df[missing_sex_mask].index:
        species = df.loc[idx, 'species']
        mass = df.loc[idx, 'body_mass_g']
        
        species_data = df[(df['species'] == species) & df['sex'].notna()]
        median_mass = species_data['body_mass_g'].median()
        
        imputed_sex = 'Male' if mass >= median_mass else 'Female'
        df.loc[idx, 'sex'] = imputed_sex
    return df

# Initialize Data
df_raw = get_raw_data()
df_clean = get_cleaned_data(df_raw)

# Sidebar layout
st.sidebar.image("https://img.icons8.com/color/96/penguin.png", width=80)
st.sidebar.markdown("### Palmer Penguins Analytics")
st.sidebar.markdown("Configure filters below to interact with the visualizations.")

# Filter Selection
species_filter = st.sidebar.multiselect(
    "Select Species",
    options=df_clean['species'].unique().tolist(),
    default=df_clean['species'].unique().tolist()
)

island_filter = st.sidebar.multiselect(
    "Select Island",
    options=df_clean['island'].unique().tolist(),
    default=df_clean['island'].unique().tolist()
)

# Apply Filters
df_filtered = df_clean[
    df_clean['species'].isin(species_filter) &
    df_clean['island'].isin(island_filter)
]

# Title
st.markdown("<div class='main-title'>Palmer Penguins Advanced Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive visualization and behavior study of Palmer Archipelago penguin populations</div>", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Profile & Cleaning",
    "📈 Research Questions",
    "🔄 Simpson's Paradox",
    "📝 Conclusions & Actions"
])

# Tab 1: Data Profile & Cleaning
with tab1:
    st.markdown("### 📊 Data Profiling and Cleaning Pipeline")
    st.markdown("""
    In raw datasets, missing values are common and must be resolved before analysis. 
    Here, we explicitly handle 19 missing values in the Palmer Penguins dataset.
    """)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df_raw)}</div><div class='metric-label'>Raw Rows</div></div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df_raw.isna().sum().sum()}</div><div class='metric-label'>Raw Nulls</div></div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df_clean)}</div><div class='metric-label'>Cleaned Rows</div></div>", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df_clean.isna().sum().sum()}</div><div class='metric-label'>Cleaned Nulls</div></div>", unsafe_allow_html=True)
        
    st.markdown("#### Imputation Strategy")
    st.info("""
    1. **Empty Rows Dropped**: Rows index 3 and 339 are completely null across all biological measurements. They carry no metrics and are discarded (reducing the count from 344 to 342).
    2. **Categorical Imputation for Sex**: Rather than discarding 9 rows with valid dimensions but missing sex identifiers, we impute them. We compare each penguin's weight to the species-specific median weight. A penguin is classified as **Male** if its body mass exceeds the species median, and **Female** otherwise. This preserves valid physical metrics.
    """)
    
    clean_data_toggle = st.radio("Display Data View", ["Cleaned Dataset (Fully Imputed)", "Raw Dataset (Contains Nulls)"], horizontal=True)
    
    if clean_data_toggle == "Cleaned Dataset (Fully Imputed)":
        st.dataframe(df_filtered, use_container_width=True)
    else:
        # Show raw data (filtered)
        raw_filtered = df_raw[
            df_raw['species'].isin(species_filter) &
            df_raw['island'].isin(island_filter)
        ]
        st.dataframe(raw_filtered, use_container_width=True)

# Tab 2: Research Questions & Visualizations
with tab2:
    st.markdown("### 📈 Visualizing Core Research Questions")
    
    # Question 1
    st.markdown("""
        <div class='rq-box'>
            <div class='rq-title'>Research Question 1: Which species is the heaviest?</div>
            <div class='rq-explanation'>We plot body mass distributions across Adelie, Chinstrap, and Gentoo species. Gentoo penguins are clearly heavier, with weight distributions shifted significantly higher than the other two species. Adelie and Chinstrap share similar weight profiles.</div>
        </div>
    """, unsafe_allow_html=True)
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_filtered, x='species', y='body_mass_g', palette='Set2', ax=ax1)
    sns.stripplot(data=df_filtered, x='species', y='body_mass_g', color='black', alpha=0.3, size=4, ax=ax1)
    ax1.set_title('Body Mass Distribution by Species')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Body Mass (g)')
    st.pyplot(fig1)
    
    # Question 2
    st.markdown("""
        <div class='rq-box'>
            <div class='rq-title'>Research Question 2: Does sex affect body mass?</div>
            <div class='rq-explanation'>By splitting the distributions by sex, we examine size differences. Across all species, male penguins are consistently heavier than their female counterparts, showing clear sexual dimorphism.</div>
        </div>
    """, unsafe_allow_html=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=df_filtered, x='species', y='body_mass_g', hue='sex', palette='coolwarm', ax=ax2)
    ax2.set_title('Body Mass Distribution by Species and Sex')
    ax2.set_xlabel('Species')
    ax2.set_ylabel('Body Mass (g)')
    st.pyplot(fig2)
    
    # Question 3 & 4
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("""
            <div class='rq-box'>
                <div class='rq-title'>Research Question 3: Which measurements are most strongly related?</div>
                <div class='rq-explanation'>The correlation heatmap displays relationships. Flipper length and body mass have a strong positive correlation (+0.87), meaning larger penguins have proportionally longer flippers. Bill depth and length show a weak negative correlation when aggregated.</div>
            </div>
        """, unsafe_allow_html=True)
        
        corr_matrix = df_filtered.select_dtypes(include='number').corr()
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", square=True, ax=ax3)
        ax3.set_title('Correlation Heatmap')
        st.pyplot(fig3)
        
    with col_v2:
        st.markdown("""
            <div class='rq-box'>
                <div class='rq-title'>Research Question 4: Are there outliers in the dataset?</div>
                <div class='rq-explanation'>We inspect boxplots of all numerical measurements. The distributions appear balanced and clean, indicating that the dataset has no extreme, distorting outliers.</div>
            </div>
        """, unsafe_allow_html=True)
        
        fig4, axes4 = plt.subplots(2, 2, figsize=(8, 7.3))
        num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        for idx, col in enumerate(num_cols):
            ax = axes4[idx // 2, idx % 2]
            sns.boxplot(data=df_filtered, y=col, color='lightblue', ax=ax)
            ax.set_title(col)
            ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig4)

# Tab 3: Simpson's Paradox
with tab3:
    st.markdown("### 🔄 Simpson's Paradox: Why Aggregation Can Mislead")
    st.markdown("""
    Simpson's Paradox occurs when a trend appearing in aggregated data reverses when the data is split into sub-groups.
    In this dataset, bill length and bill depth show this exact statistical anomaly.
    """)
    
    paradox_toggle = st.radio(
        "Select Trend Visualization Mode",
        ["Show Aggregated Trend (Negative Correlation)", "Show Species-Specific Trends (Positive Correlation)"],
        horizontal=True
    )
    
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    
    if paradox_toggle == "Show Aggregated Trend (Negative Correlation)":
        sns.regplot(data=df_filtered, x='bill_length_mm', y='bill_depth_mm', ax=ax5, color='darkred', scatter_kws={'alpha':0.5})
        ax5.set_title("Aggregated View: Bill Length vs Bill Depth (r = -0.23)")
        st.markdown("""
        > [!WARNING]
        > Looking at the aggregated data, bill length and bill depth appear to have a negative correlation.
        > This suggests that longer bills are shallower, which is biologically counter-intuitive.
        """)
    else:
        colors = sns.color_palette()
        species_colors = {
            'Adelie': colors[0],
            'Chinstrap': colors[1],
            'Gentoo': colors[2]
        }
        sns.scatterplot(data=df_filtered, x='bill_length_mm', y='bill_depth_mm', hue='species', style='species', s=70, palette=species_colors, ax=ax5)
        for sp in df_filtered['species'].unique():
            sp_df = df_filtered[df_filtered['species'] == sp]
            if len(sp_df) > 1:
                sns.regplot(data=sp_df, x='bill_length_mm', y='bill_depth_mm', scatter=False, color=species_colors[sp], ax=ax5)
        ax5.set_title("Species-Specific View: Bill Length vs Bill Depth (Simpson's Paradox)")
        st.markdown("""
        > [!TIP]
        > When we look at each species individually, the trend reverses!
        > Within every species, bill length and bill depth are positively correlated.
        > This paradox occurs because Gentoo penguins have long, shallow bills, while Adelie penguins have short, deep bills.
        """)
        
    ax5.set_xlabel('Bill Length (mm)')
    ax5.set_ylabel('Bill Depth (mm)')
    st.pyplot(fig5)

# Tab 4: Conclusions & Actions
with tab4:
    st.markdown("### 📝 Executive Summary and Analytical Insights")
    
    st.markdown("""
    #### 1. Core Biological Insights
    * **Species is the Primary Size Driver**: The Gentoo species is significantly larger and heavier than both Adelie and Chinstrap penguins (with an average mass exceeding 5000g compared to ~3700g). 
    * **Pronounced Sexual Dimorphism**: Within every single species, male penguins are consistently heavier than female penguins by 10% to 15%. This size difference shows that sex is a major driver of body mass within species.
    
    #### 2. The Analytical Pitfall: Simpson's Paradox
    * **The Danger of Aggregated Statistics**: Looking at the aggregated data, bill length and bill depth appear to have a negative correlation (-0.23). However, when we divide the data by species, they show a strong positive correlation within each species.
    * **Why it Occurs**: Gentoo penguins are larger but have relatively long, shallow bills. Adelie penguins are smaller but have shorter, deep bills. If you do not account for species sub-groups, the aggregated trend leads to a false biological conclusion.
    
    #### 3. Key Recommendations for Data Analysts and Ecologists
    * **Segment Before Correlating**: When analyzing biological, ecological, or demographic datasets containing distinct subgroups, always segment the analysis. Aggregating data across distinct subgroups can lead to misleading statistical outcomes (Simpson's Paradox).
    * **Control for Confounders in Modeling**: When building classifiers or predictive models (e.g. predicting species or sex based on dimensions), species and sex must be controlled for. They are powerful confounding variables that completely structure the scaling relationships of the body measurements.
    """)
