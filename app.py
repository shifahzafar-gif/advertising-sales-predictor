import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Sales Predictor Pro", page_icon="📈", layout="wide")

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    /* Main background gradient and fonts */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
        color: #0f172a !important;
    }
    
    /* Force text color for all standard text elements */
    p, span, div, li, label {
        color: #0f172a !important;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a8a !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        color: #059669 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        color: #475569 !important;
    }
    
    /* Card-like containers for sidebar and main areas */
    .css-1d391kg, .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 20px;
    }
    
    /* Button styling (if we add any buttons) */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('advertising_data.csv')
        return df
    except Exception as e:
        return None

@st.cache_resource
def train_model(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    return model, r2, mse

def main():
    st.title("📈 Advertising Sales Predictor")
    st.markdown("Predict your future sales based on advertising budget allocation across **TV**, **Radio**, and **Newspaper** channels using a machine learning model.")

    df = load_data()
    
    if df is None:
        st.error("Dataset not found! Please ensure 'advertising 2(in).csv' is located at '/Users/shifahzafar/Downloads/'.")
        return
        
    model, r2, mse = train_model(df)

    # --- User Input ---
    st.header("🎯 Budget Allocation")
    st.markdown("Adjust the sliders below to see real-time sales predictions.")
    
    col_tv, col_radio, col_news = st.columns(3)
    
    with col_tv:
        tv_budget = st.slider("TV Advertising ($K)", 
                              min_value=float(df['TV'].min()), 
                              max_value=float(df['TV'].max()), 
                              value=float(df['TV'].mean()))
                              
    with col_radio:
        radio_budget = st.slider("Radio Advertising ($K)", 
                                 min_value=float(df['Radio'].min()), 
                                 max_value=float(df['Radio'].max()), 
                                 value=float(df['Radio'].mean()))
                                 
    with col_news:
        news_budget = st.slider("Newspaper Advertising ($K)", 
                                min_value=float(df['Newspaper'].min()), 
                                max_value=float(df['Newspaper'].max()), 
                                value=float(df['Newspaper'].mean()))
                                
    st.divider()
    
    # --- Main Content Area ---
    col1, col2 = st.columns([1.5, 1])
    
    with col2:
        st.subheader("📊 Sales Prediction")
        # Predict based on input
        input_data = pd.DataFrame({'TV': [tv_budget], 'Radio': [radio_budget], 'Newspaper': [news_budget]})
        prediction = model.predict(input_data)[0]
        
        st.metric(label="Estimated Sales Units", value=f"{prediction:.2f} k", delta="Model Output")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Model Performance:**\n\n- **R² Score**: {r2:.2f}\n- **MSE**: {mse:.2f}")
        st.success("The algorithm uses a Random Forest Regressor trained on historical advertising data.")
        
    with col1:
        st.subheader("💡 Dataset Overview")
        st.markdown("Here is a quick glance at the historical advertising data:")
        st.dataframe(df.head(10), width='stretch')
        
        st.subheader("🔍 Budget vs Sales Insights")
        st.markdown("""
        - **TV Advertising** typically shows the strongest correlation with sales.
        - **Radio Advertising** consistently provides solid complementary reach.
        - **Newspaper Advertising** often has the lowest individual impact.
        """)
        
if __name__ == "__main__":
    main()
