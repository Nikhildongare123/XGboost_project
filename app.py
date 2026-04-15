import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: semi-bold;
        color: #2C5F8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background-color: #1E3A5F;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-text {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        margin: 0;
    }
    .info-text {
        font-size: 1rem;
        color: #666;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# File paths
MODEL_PATH = SCRIPT_DIR / 'model (2).pkl'
DATA_PATH = SCRIPT_DIR / 'Concrete_Data_Yeh.csv'

# Alternative paths if files are in different locations
# You can also specify absolute paths like:
# MODEL_PATH = Path('/path/to/your/model (2).pkl')
# DATA_PATH = Path('/path/to/your/Concrete_Data_Yeh.csv')

# Load the model
@st.cache_resource
def load_model():
    try:
        if not MODEL_PATH.exists():
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.info("Please ensure 'model (2).pkl' is in the same directory as this script.")
            return None
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load and prepare data
@st.cache_data
def load_data():
    try:
        if not DATA_PATH.exists():
            st.error(f"Data file not found at: {DATA_PATH}")
            st.info("Please ensure 'Concrete_Data_Yeh.csv' is in the same directory as this script.")
            return None
        
        df = pd.read_csv(DATA_PATH)
        st.success(f"✅ Data loaded successfully! ({len(df)} records)")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Feature names
FEATURES = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 
            'coarseaggregate', 'fineaggregate', 'age']

FEATURE_NAMES_DISPLAY = {
    'cement': 'Cement (kg/m³)',
    'slag': 'Blast Furnace Slag (kg/m³)',
    'flyash': 'Fly Ash (kg/m³)',
    'water': 'Water (kg/m³)',
    'superplasticizer': 'Superplasticizer (kg/m³)',
    'coarseaggregate': 'Coarse Aggregate (kg/m³)',
    'fineaggregate': 'Fine Aggregate (kg/m³)',
    'age': 'Age (days)'
}

# Default values for each feature (median from dataset)
DEFAULT_VALUES = {
    'cement': 305.0,
    'slag': 0.0,
    'flyash': 0.0,
    'water': 185.0,
    'superplasticizer': 6.0,
    'coarseaggregate': 1050.0,
    'fineaggregate': 800.0,
    'age': 28.0
}

# Min and max values from data (for sliders)
MIN_MAX_VALUES = {
    'cement': (100, 550),
    'slag': (0, 360),
    'flyash': (0, 200),
    'water': (120, 230),
    'superplasticizer': (0, 35),
    'coarseaggregate': (800, 1150),
    'fineaggregate': (600, 1000),
    'age': (1, 365)
}

def predict_strength(model, input_data):
    """Make prediction using the loaded model"""
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return prediction

def get_strength_category(strength):
    """Categorize concrete strength"""
    if strength < 20:
        return "Low Strength", "🔴"
    elif strength < 40:
        return "Medium Strength", "🟡"
    elif strength < 60:
        return "High Strength", "🟢"
    else:
        return "Very High Strength", "💪"

# File upload section for manual file selection
def file_upload_section():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 File Upload")
    
    uploaded_model = st.sidebar.file_uploader("Upload Model File (optional)", type=['pkl'])
    uploaded_data = st.sidebar.file_uploader("Upload Data File (optional)", type=['csv'])
    
    return uploaded_model, uploaded_data

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🏗️ Concrete Compressive Strength Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="info-text">Predict the 28-day compressive strength of concrete using mixture proportions</div>', 
                unsafe_allow_html=True)
    
    # File upload section
    uploaded_model, uploaded_data = file_upload_section()
    
    # Load model and data
    model = None
    df = None
    
    if uploaded_model is not None:
        # Use uploaded model
        try:
            model = pickle.load(uploaded_model)
            st.sidebar.success("✅ Custom model loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading custom model: {e}")
    else:
        # Try to load from default path
        model = load_model()
    
    if uploaded_data is not None:
        # Use uploaded data
        try:
            df = pd.read_csv(uploaded_data)
            st.sidebar.success(f"✅ Custom data loaded! ({len(df)} records)")
        except Exception as e:
            st.sidebar.error(f"Error loading custom data: {e}")
    else:
        # Try to load from default path
        df = load_data()
    
    if model is None:
        st.warning("""
        ### ⚠️ Model Not Loaded
        
        Please ensure the model file 'model (2).pkl' is in the same directory as this script.
        
        **Troubleshooting steps:**
        1. Check if the file exists in the current directory
        2. Use the file uploader in the sidebar to upload the model manually
        3. Verify the file name is exactly 'model (2).pkl' (including spaces)
        
        **Current working directory:** `{}`
        """.format(os.getcwd()))
        st.stop()
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("## 📊 Input Parameters")
        st.markdown("Adjust the concrete mixture proportions:")
        
        input_data = {}
        
        # Create input widgets for each feature
        for feature in FEATURES:
            default_val = DEFAULT_VALUES[feature]
            min_val, max_val = MIN_MAX_VALUES[feature]
            
            if feature == 'age':
                input_data[feature] = st.slider(
                    FEATURE_NAMES_DISPLAY[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=1.0,
                    help="Age of concrete in days (1-365)"
                )
            else:
                input_data[feature] = st.number_input(
                    FEATURE_NAMES_DISPLAY[feature],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=5.0,
                    format="%.1f"
                )
        
        # Reset button
        if st.button("🔄 Reset to Default Values", use_container_width=True):
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📐 Current Mixture Composition</div>', 
                    unsafe_allow_html=True)
        
        # Create a DataFrame for display
        display_data = []
        for feature in FEATURES:
            display_data.append({
                'Component': FEATURE_NAMES_DISPLAY[feature],
                'Value': f"{input_data[feature]:.1f}",
                'Unit': 'kg/m³' if feature != 'age' else 'days'
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Feature importance plot (if we have data)
        if df is not None:
            st.markdown('<div class="sub-header">📈 Data Overview</div>', 
                        unsafe_allow_html=True)
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Strength Range", f"{df['csMPa'].min():.1f} - {df['csMPa'].max():.1f} MPa")
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Results</div>', 
                    unsafe_allow_html=True)
        
        # Make prediction
        prediction = predict_strength(model, input_data)
        category, emoji = get_strength_category(prediction)
        
        # Display prediction
        st.markdown(f"""
            <div class="prediction-box">
                <p style="color: white; margin-bottom: 5px;">Predicted Compressive Strength</p>
                <p class="prediction-text">{prediction:.1f} MPa</p>
                <p style="color: white; margin-top: 5px;">{emoji} {category}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional info based on strength
        if prediction < 20:
            st.info("⚠️ **Note:** This strength is relatively low. Consider increasing cement content or reducing water-cement ratio.")
        elif prediction < 40:
            st.info("📌 **Note:** This is a standard strength suitable for general construction purposes.")
        elif prediction < 60:
            st.success("✅ **Note:** This is high-strength concrete, suitable for structural applications.")
        else:
            st.success("🏆 **Note:** Very high-strength concrete! Suitable for specialized high-performance applications.")
    
    # Visualization section (only if data is available)
    if df is not None:
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Analysis & Visualization</div>', 
                    unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📈 Strength Distribution", "🔬 Feature Correlation", "📉 Prediction Analysis"])
        
        with tab1:
            # Strength distribution histogram
            fig = px.histogram(df, x='csMPa', nbins=50, 
                              title='Distribution of Concrete Compressive Strengths in Dataset',
                              labels={'csMPa': 'Compressive Strength (MPa)'},
                              color_discrete_sequence=['#1E3A5F'])
            fig.add_vline(x=prediction, line_dash="dash", line_color="red",
                         annotation_text=f"Your Prediction: {prediction:.1f} MPa")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Strength", f"{df['csMPa'].mean():.1f} MPa")
            with col2:
                st.metric("Median Strength", f"{df['csMPa'].median():.1f} MPa")
            with col3:
                st.metric("Std Deviation", f"{df['csMPa'].std():.1f} MPa")
            with col4:
                percentile = (df['csMPa'] < prediction).mean() * 100
                st.metric("Your Strength Percentile", f"{percentile:.1f}%")
        
        with tab2:
            # Correlation heatmap
            corr_matrix = df[FEATURES + ['csMPa']].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix",
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Insight:** Features with high positive correlation (red) to strength are beneficial, while negative correlation (blue) indicates higher values may reduce strength.")
        
        with tab3:
            # Sample similar mixtures
            st.markdown("#### 🔍 Similar Mixtures in Dataset")
            
            # Calculate similarity based on Euclidean distance
            input_array = np.array([input_data[f] for f in FEATURES])
            df_features = df[FEATURES].values
            distances = np.sqrt(((df_features - input_array) ** 2).sum(axis=1))
            similar_indices = distances.argsort()[:10]
            
            similar_mixtures = df.iloc[similar_indices][FEATURES + ['csMPa']].copy()
            similar_mixtures['Similarity Score'] = (1 - distances[similar_indices] / distances[similar_indices].max()) * 100
            
            st.dataframe(
                similar_mixtures.style.format({
                    'cement': '{:.0f}',
                    'slag': '{:.0f}',
                    'flyash': '{:.0f}',
                    'water': '{:.0f}',
                    'superplasticizer': '{:.1f}',
                    'coarseaggregate': '{:.0f}',
                    'fineaggregate': '{:.0f}',
                    'age': '{:.0f}',
                    'csMPa': '{:.1f}',
                    'Similarity Score': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("Showing the 10 most similar mixtures from the training dataset based on mixture proportions.")
    else:
        st.info("📊 Upload the data file to see visualizations and analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>⚠️ <strong>Disclaimer:</strong> This prediction is based on a machine learning model trained on specific concrete mixture data. 
            Actual concrete strength may vary based on curing conditions, material quality, and other factors not captured in this model.</p>
            <p>🏗️ Built with Streamlit • Model: XGBoost Regressor</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
