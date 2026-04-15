import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from pathlib import Path
import io

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

# Default values for each feature
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

# Min and max values
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

# Sample data for demonstration (if real data can't be loaded)
SAMPLE_DATA = pd.DataFrame({
    'cement': [540, 540, 332.5, 332.5, 198.6],
    'slag': [0, 0, 142.5, 142.5, 132.4],
    'flyash': [0, 0, 0, 0, 0],
    'water': [162, 162, 228, 228, 192],
    'superplasticizer': [2.5, 2.5, 0, 0, 0],
    'coarseaggregate': [1040, 1055, 932, 932, 978.4],
    'fineaggregate': [676, 676, 594, 594, 825.5],
    'age': [28, 28, 270, 365, 360],
    'csMPa': [79.99, 61.89, 40.27, 41.05, 44.3]
})

def load_model_from_file():
    """Try to load model from various possible locations"""
    possible_paths = [
        Path('model (2).pkl'),
        Path('/mount/src/xgboost_project/model (2).pkl'),
        Path('./model (2).pkl'),
        Path('model.pkl'),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                return model, path
            except:
                continue
    return None, None

def load_data_from_file():
    """Try to load data from various possible locations"""
    possible_paths = [
        Path('Concrete_Data_Yeh.csv'),
        Path('/mount/src/xgboost_project/Concrete_Data_Yeh.csv'),
        Path('./Concrete_Data_Yeh.csv'),
        Path('data.csv'),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df, path
            except:
                continue
    return None, None

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

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🏗️ Concrete Compressive Strength Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="info-text">Predict the compressive strength of concrete using mixture proportions</div>', 
                unsafe_allow_html=True)
    
    # Initialize session state for model and data
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("## 📁 File Upload")
        st.markdown("Upload your model and data files:")
        
        # Model upload
        uploaded_model = st.file_uploader(
            "Upload Model File (model.pkl)", 
            type=['pkl'],
            help="Upload the trained XGBoost model file"
        )
        
        # Data upload
        uploaded_data = st.file_uploader(
            "Upload Data File (CSV)", 
            type=['csv'],
            help="Upload the concrete data CSV file"
        )
        
        if uploaded_model is not None:
            try:
                st.session_state.model = pickle.load(uploaded_model)
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
        
        if uploaded_data is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_data)
                st.success(f"✅ Data loaded! ({len(st.session_state.df)} records)")
            except Exception as e:
                st.error(f"Error loading data: {e}")
        
        # Try to load from files if not uploaded
        if st.session_state.model is None:
            st.markdown("---")
            st.markdown("### 🔍 Trying default files...")
            model, model_path = load_model_from_file()
            if model is not None:
                st.session_state.model = model
                st.success(f"✅ Model loaded from: {model_path}")
            else:
                st.warning("⚠️ No model file found. Please upload a model file.")
        
        if st.session_state.df is None:
            df, data_path = load_data_from_file()
            if df is not None:
                st.session_state.df = df
                st.success(f"✅ Data loaded from: {data_path}")
        
        st.markdown("---")
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
                    help="Age of concrete in days"
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
    
    # Check if model is loaded
    if st.session_state.model is None:
        st.warning("""
        ### ⚠️ Model Not Loaded
        
        Please upload the model file 'model (2).pkl' using the file uploader in the sidebar.
        
        **Instructions:**
        1. Click on "Browse files" in the sidebar under "Upload Model File"
        2. Select your 'model (2).pkl' file
        3. Wait for the model to load
        
        **Note:** The model will be loaded into memory and used for predictions.
        """)
        
        # Show demo mode option
        if st.button("🎯 Try Demo Mode (Sample Predictions)"):
            st.session_state.demo_mode = True
            st.rerun()
        st.stop()
    
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
        
        # Data overview if available
        if st.session_state.df is not None:
            st.markdown('<div class="sub-header">📈 Data Overview</div>', 
                        unsafe_allow_html=True)
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Total Records", f"{len(st.session_state.df):,}")
            with col1b:
                st.metric("Features", f"{len(FEATURES)}")
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Prediction Results</div>', 
                    unsafe_allow_html=True)
        
        # Make prediction
        prediction = predict_strength(st.session_state.model, input_data)
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
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Analysis & Visualization</div>', 
                    unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📈 Strength Distribution", "🔬 Feature Correlation", "📉 Similar Mixtures"])
        
        with tab1:
            # Strength distribution histogram
            fig = px.histogram(st.session_state.df, x='csMPa', nbins=50, 
                              title='Distribution of Concrete Compressive Strengths in Dataset',
                              labels={'csMPa': 'Compressive Strength (MPa)'},
                              color_discrete_sequence=['#1E3A5F'])
            fig.add_vline(x=prediction, line_dash="dash", line_color="red",
                         annotation_text=f"Your Prediction: {prediction:.1f} MPa")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1a, col1b, col1c, col1d = st.columns(4)
            with col1a:
                st.metric("Mean Strength", f"{st.session_state.df['csMPa'].mean():.1f} MPa")
            with col1b:
                st.metric("Median Strength", f"{st.session_state.df['csMPa'].median():.1f} MPa")
            with col1c:
                st.metric("Std Deviation", f"{st.session_state.df['csMPa'].std():.1f} MPa")
            with col1d:
                percentile = (st.session_state.df['csMPa'] < prediction).mean() * 100
                st.metric("Percentile", f"{percentile:.1f}%")
        
        with tab2:
            if len(st.session_state.df) > 10:
                # Correlation heatmap
                corr_matrix = st.session_state.df[FEATURES + ['csMPa']].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Feature Correlation Matrix",
                               color_continuous_scale='RdBu_r',
                               zmin=-1, zmax=1)
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Insight:** Features with high positive correlation (red) to strength are beneficial, while negative correlation (blue) indicates higher values may reduce strength.")
            else:
                st.info("Need more data points for correlation analysis.")
        
        with tab3:
            # Sample similar mixtures
            st.markdown("#### 🔍 Similar Mixtures in Dataset")
            
            # Calculate similarity based on Euclidean distance
            input_array = np.array([input_data[f] for f in FEATURES])
            df_features = st.session_state.df[FEATURES].values
            distances = np.sqrt(((df_features - input_array) ** 2).sum(axis=1))
            similar_indices = distances.argsort()[:10]
            
            similar_mixtures = st.session_state.df.iloc[similar_indices][FEATURES + ['csMPa']].copy()
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
            
            st.caption("Showing the 10 most similar mixtures from the training dataset.")
    
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
