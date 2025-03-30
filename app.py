import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import joblib
import os

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSidebar {
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
    }
    .stNumberInput input, .stTextInput input {
        background-color: #ecf0f1 !important;
        color: #2c3e50 !important;
        border: 1px solid #bdc3c7 !important;
        border-radius: 5px !important;
        padding: 8px 12px !important;
    }
    .stNumberInput label, .stTextInput label {
        color: #ecf0f1 !important;
        font-weight: 500 !important;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .input-description {
        color: #ecf0f1;
        font-size: 12px;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .sidebar-header {
        color: #ecf0f1;
        font-size: 1.5rem;
        margin-bottom: 20px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .feature-input-section {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Constants
MODEL_PATH = 'machine_failure_model.pkl'
DATA_FILE = 'machine.csv'  # Local file in the same directory

def load_data():
    """Load data from local file with error handling"""
    try:
        if not os.path.exists(DATA_FILE):
            st.error(f"Data file '{DATA_FILE}' not found. Please ensure it's in the same directory as this app.")
            st.stop()
        return pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        st.stop()

def train_and_save_model():
    """Train the model and save to disk"""
    df = load_data()
    
    # Preprocessing
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    
    # Prepare features and target
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save model
    model_data = {
        'model': model,
        'features': X.columns.tolist(),
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(model_data, MODEL_PATH)
    return model_data

def load_model():
    """Load model from disk or train if not exists"""
    if not os.path.exists(MODEL_PATH):
        st.info("No pre-trained model found. Training a new model...")
        return train_and_save_model()
    
    try:
        model_data = joblib.load(MODEL_PATH)
        st.success("Loaded pre-trained model successfully!")
        return model_data
    except Exception as e:
        st.warning(f"Error loading model: {str(e)}. Training a new model...")
        return train_and_save_model()

# Load or train model
model_data = load_model()
model = model_data['model']
features = model_data['features']
X_test = model_data['X_test']
y_test = model_data['y_test']

# Sidebar for user input
st.sidebar.markdown('<div class="sidebar-header">Machine Parameters</div>', unsafe_allow_html=True)

# Feature information - now with suggested ranges in description
feature_info = {
    'Air temperature [K]': {
        'desc': 'Ambient air temperature (295-315 K)',
        'default': 300.0,
        'step': 0.1,
        'unit': 'K'
    },
    'Process temperature [K]': {
        'desc': 'Process temperature (305-315 K)',
        'default': 310.0,
        'step': 0.1,
        'unit': 'K'
    },
    'Rotational speed [rpm]': {
        'desc': 'Rotational speed (1000-3000 rpm)',
        'default': 1500,
        'step': 10,
        'unit': 'rpm'
    },
    'Torque [Nm]': {
        'desc': 'Rotational force (10-80 Nm)',
        'default': 40.0,
        'step': 0.5,
        'unit': 'Nm'
    },
    'Tool wear [min]': {
        'desc': 'Tool wear time (0-300 min)',
        'default': 50,
        'step': 1,
        'unit': 'min'
    }
}

# Collect user input - all as number input fields
user_input = {}
for feature in features:
    if feature in feature_info:
        info = feature_info[feature]
        st.sidebar.markdown(f"**{feature}**")
        st.sidebar.markdown(f'<div class="input-description">{info["desc"]}</div>', unsafe_allow_html=True)
        user_input[feature] = st.sidebar.number_input(
            label=f"Enter value ({info['unit']})",
            value=info['default'],
            step=info['step'],
            key=feature,
            format="%.1f" if isinstance(info['default'], float) else "%d"
        )
    else:
        # Handle other features (like one-hot encoded ones)
        user_input[feature] = 0.0  # Default value for other features

# Prediction button
if st.sidebar.button("Predict Failure Risk", key="predict_button"):
    # Prepare input data
    input_data = [user_input[feature] for feature in features]
    
    # Make prediction
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]
    
    # Display results
    st.markdown('<div class="card"><h3>Prediction Result</h3></div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of Failure (Probability: {probability:.1%})")
        st.markdown("""
        <div style="background-color: #fdecea; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <p style="margin: 0;">Recommend immediate maintenance inspection. Parameters indicate potential failure conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Normal Operation (Probability: {1-probability:.1%})")
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <p style="margin: 0;">Machine operating within normal parameters. Continue routine monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show input parameters
    st.subheader("Input Parameters")
    cols = st.columns(2)
    for i, (feature, value) in enumerate(user_input.items()):
        if feature in feature_info:
            with cols[i % 2]:
                st.markdown(f"**{feature}**: {value} {feature_info[feature]['unit']}")

# Main page content
st.title("Machine Failure Prediction System")
st.markdown("""
<div class="card">
    <p>This system predicts potential machine failures using a Random Forest classifier trained on historical equipment data.</p>
    <p>Enter the current machine parameters in the sidebar and click <strong>Predict Failure Risk</strong> to get an assessment.</p>
</div>
""", unsafe_allow_html=True)

# Data sample - using the local file
with st.expander("View Sample Data"):
    try:
        sample_data = pd.read_csv(DATA_FILE).head(5)
        st.dataframe(sample_data)
    except Exception as e:
        st.error(f"Could not load sample data: {str(e)}")
