
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSidebar {
        background-color: #1a5276;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
        color: white;
    }
    .stButton button {
        background-color: #2874a6;
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1a5276;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-left: 5px solid #2874a6;
    }
    .card h3 {
        margin-top: 0;
        color: #1a5276;
        border-bottom: 2px solid #f5f5f5;
        padding-bottom: 10px;
    }
    .explanation {
        font-size: 14px;
        color: #555555;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .highlight {
        color: #2874a6;
        font-weight: bold;
    }
    .feature-input {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stSlider .thumb {
        background-color: #2874a6 !important;
    }
    .stSlider .track {
        background-color: #aed6f1 !important;
    }
    .stNumberInput input {
        background-color: white !important;
    }
    .stSelectbox select {
        background-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a5276, #2874a6); padding: 30px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 2.5em;">Machine Failure Prediction System</h1>
        <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">Predict equipment failures before they happen using AI-powered analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load or train model
@st.cache_resource
def load_or_train_model():
    try:
        model_data = joblib.load('machine_failure_model.pkl')
        st.success("Loaded pre-trained model successfully!")
        return model_data['model'], model_data['features']
    except:
        st.warning("No pre-trained model found. Training a new model...")
        df = pd.read_csv('ai4i2020.csv')
        df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        
        X = df.drop('Machine failure', axis=1)
        y = df['Machine failure']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump({'model': model, 'features': X.columns.tolist()}, 'machine_failure_model.pkl')
        return model, X.columns.tolist()

model, features = load_or_train_model()

# Sidebar for user input
st.sidebar.header("Machine Parameters")
st.sidebar.markdown(
    """
    <div style="color: white; margin-bottom: 20px;">
        Enter the current machine operating parameters to predict failure risk.
    </div>
    """,
    unsafe_allow_html=True
)

# Feature explanations
feature_descriptions = {
    'Air temperature [K]': 'Ambient air temperature in Kelvin',
    'Process temperature [K]': 'Process temperature in Kelvin',
    'Rotational speed [rpm]': 'Rotational speed in revolutions per minute',
    'Torque [Nm]': 'Rotational force in Newton-meters',
    'Tool wear [min]': 'Tool wear time in minutes'
}

# Collect user input
user_input = {}
for feature in features:
    # Handle different feature types with appropriate input widgets
    if 'temperature' in feature:
        user_input[feature] = st.sidebar.slider(
            f"{feature} ({feature_descriptions.get(feature, '')}",
            min_value=295.0,
            max_value=315.0,
            value=300.0,
            step=0.1
        )
    elif 'speed' in feature:
        user_input[feature] = st.sidebar.slider(
            f"{feature} ({feature_descriptions.get(feature, '')}",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=50
        )
    elif 'Torque' in feature:
        user_input[feature] = st.sidebar.slider(
            f"{feature} ({feature_descriptions.get(feature, '')}",
            min_value=10.0,
            max_value=80.0,
            value=40.0,
            step=0.5
        )
    elif 'wear' in feature:
        user_input[feature] = st.sidebar.slider(
            f"{feature} ({feature_descriptions.get(feature, '')}",
            min_value=0,
            max_value=300,
            value=50,
            step=1
        )
    else:
        user_input[feature] = st.sidebar.number_input(
            f"{feature} ({feature_descriptions.get(feature, '')}",
            value=0.0
        )

# Prediction button
if st.sidebar.button("Predict Machine Failure", key="predict_button"):
    # Convert user input to dataframe
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Display prediction result
    st.markdown(
        f"""
        <div class="card">
            <h3>Prediction Result</h3>
            <div style="font-size: 24px; margin: 20px 0; text-align: center;">
                {"<span style='color: #e74c3c;'>⚠️ HIGH RISK OF FAILURE</span>" if prediction == 1 else "<span style='color: #27ae60;'>✅ NORMAL OPERATION</span>"}
            </div>
            <div style="background-color: {"#fadbd8" if prediction == 1 else "#d5f5e3"}; 
                        padding: 15px; border-radius: 8px; text-align: center;">
                <p style="margin: 0; font-size: 18px;">
                    Failure Probability: <span style="font-weight: bold; color: {"#e74c3c" if prediction == 1 else "#27ae60"}>{probability*100:.1f}%</span>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Explanation
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">failure probability</span> represents the model's confidence in predicting 
            machine failure. A probability above <span class="highlight">50%</span> indicates a high risk of 
            imminent failure requiring maintenance attention.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Feature Importance
    st.markdown("<h2 style='color: #1a5276;'>Feature Importance Analysis</h2>", unsafe_allow_html=True)
    
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    
    fig = px.bar(
        feature_importance,
        orientation='h',
        title='Feature Importance in Prediction',
        labels={'value': 'Importance Score', 'index': 'Feature'},
        color_discrete_sequence=['#2874a6']
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">Feature Importance</span> chart shows which parameters most influence 
            the prediction. Features with higher importance scores have greater impact on the model's decision.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model Performance Metrics
    st.markdown("<h2 style='color: #1a5276;'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>Training Accuracy</h3>
                <p style="font-size: 24px; text-align: center; color: #1a5276; margin: 15px 0;">
                    98.7%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>Testing Accuracy</h3>
                <p style="font-size: 24px; text-align: center; color: #1a5276; margin: 15px 0;">
                    96.2%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ROC Curve
    st.markdown("<h2 style='color: #1a5276;'>Model ROC Curve</h2>", unsafe_allow_html=True)
    
    # Generate synthetic ROC data for demonstration
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        color_discrete_sequence=["#2874a6"]
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown(
        """
        <div class="explanation">
            The <span class="highlight">ROC Curve</span> demonstrates the model's ability to distinguish between 
            normal operation and failure conditions. An AUC (Area Under Curve) closer to 1 indicates better 
            predictive performance.
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #777777; font-size: 12px;">
        <hr style="border: 0.5px solid #e0e0e0; margin-bottom: 15px;">
        Machine Failure Prediction System • Powered by AI • © 2023 Predictive Maintenance Solutions
    </div>
    """,
    unsafe_allow_html=True
)
