import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_curve, auc, 
                           confusion_matrix, classification_report)
import joblib
import os
from imblearn.over_sampling import SMOTE

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
 
# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
       background-color: #2f2f2f;
        color: white;
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
        background-color: #00008B;
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
        color: black;
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
DATA_FILE = 'machine.csv'

# Hero Section
st.markdown(
    """
    <div style="background-color:#192724; padding: 20px; border-radius: 10px; color: white;">
        <h1 style="margin: 0;">Machine Failure Prediction System</h1>
        <p style="margin: 0;">Predict equipment failures before they happen using advanced machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

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
    
    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                             cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Save model
    model_data = {
        'model': best_rf,
        'features': X.columns.tolist(),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'best_params': grid_search.best_params_
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
        
        # Backward compatibility check
        required_keys = ['model', 'features', 'X_test', 'y_test']
        if not all(key in model_data for key in required_keys):
            st.warning("Old model format detected. Retraining new model...")
            return train_and_save_model()
            
        return model_data
    except Exception as e:
        st.warning(f"Error loading model: {str(e)}. Training a new model...")
        return train_and_save_model()

# Load or train model
model_data = load_model()
model = model_data['model']
features = model_data['features']
X_train = model_data.get('X_train', None)
y_train = model_data.get('y_train', None)
X_test = model_data.get('X_test', None)
y_test = model_data.get('y_test', None)
best_params = model_data.get('best_params', {})

# Sidebar for user input
st.sidebar.markdown('<div class="sidebar-header">Machine Parameters</div>', unsafe_allow_html=True)
st.sidebar.write("Enter current operating conditions:")

# Feature information
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
            label=f"Value in {info['unit']}",
            value=info['default'],
            step=info['step'],
            key=feature,
            format="%.1f" if isinstance(info['default'], float) else "%d"
        )
    else:
        user_input[feature] = 0.0

# Prediction button
if st.sidebar.button("Predict Failure Risk"):
    # Prepare input data
    input_data = [user_input[feature] for feature in features]
    
    # Make prediction
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]
    
    # Display prediction result
    st.markdown(
        f"""
        <div class="card">
            <h3>Prediction: {'Machine WILL Fail ⚠️' if prediction == 1 else 'Machine Will NOT Fail ✅'}</h3>
            <p>Confidence Score: {probability:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Explanation for Confidence Score
    st.markdown(
        """
        <div class="explanation" style="color: white;">
            The <span class="highlight">Confidence Score</span> represents the model's certainty in its prediction. 
            A score closer to <span class="highlight">1</span> indicates high confidence in failure prediction.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Feature Importance Plot
    st.markdown("<h2 style='color: white;'>Feature Importance</h2>", unsafe_allow_html=True)
    feature_importances = model.feature_importances_
    fig_feature_importance = px.bar(
        x=features, 
        y=feature_importances, 
        labels={'x': 'Features', 'y': 'Importance'}, 
        title="Feature Importance"
    )
    st.plotly_chart(fig_feature_importance)

    # Confusion Matrix (if test data available)
    if X_test is not None and y_test is not None:
        predictions_val = model.predict(X_test)
        
        # Confusion Matrix
        st.markdown("<h2 style='color: white;'>Confusion Matrix</h2>", unsafe_allow_html=True)
        conf_matrix = confusion_matrix(y_test, predictions_val)
        fig_conf_matrix = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Failure", "Failure"],
            y=["No Failure", "Failure"],
            title="Confusion Matrix (Test Data)"
        )
        st.plotly_chart(fig_conf_matrix)

        # ROC Curve
        st.markdown("<h2 style='color: white;'>ROC Curve</h2>", unsafe_allow_html=True)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        roc_auc = auc(fpr, tpr)
        fig_roc = px.line(
            x=fpr, 
            y=tpr, 
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc)

        # Model Performance
        st.markdown("<h2 style='color: white;'>Model Performance</h2>", unsafe_allow_html=True)
        
        if X_test is not None and y_test is not None:
            predictions_val = model.predict(X_test)
            accuracy_test = accuracy_score(y_test, predictions_val)
            
            # Create columns for metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                if X_train is not None and y_train is not None:
                    accuracy_train = accuracy_score(y_train, model.predict(X_train))
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>Training Accuracy</h3>
                            <p>{accuracy_train:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="card">
                            <h3>Training Accuracy</h3>
                            <p>N/A</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>Test Accuracy</h3>
                        <p>{accuracy_test:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="card">
                    <h3>Model Performance</h3>
                    <p>Test data not available for evaluation</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Classification Report
        st.markdown("<h2 style='color: white;'>Classification Report</h2>", unsafe_allow_html=True)
        report = classification_report(y_test, predictions_val, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

    # Best Parameters
    if best_params:
        st.markdown("<h2 style='color: white;'>Optimal Model Parameters</h2>", unsafe_allow_html=True)
        st.json(best_params)

# Main page content
st.markdown("""
<div class="card">
    <p>This system predicts potential machine failures using a Random Forest classifier trained on historical equipment data.</p>
    <p>Enter the current machine parameters in the sidebar and click <strong>Predict Failure Risk</strong> to get an assessment.</p>
</div>
""", unsafe_allow_html=True)

# Model information
with st.expander("Model Details"):
    st.markdown("""
    **Algorithm**: Random Forest Classifier
    
    **Key Features**:
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear
    
    **Training Samples**: {:,}
    **Test Samples**: {:,}
    """.format(
        len(X_train) if X_train is not None else 0,
        len(X_test) if X_test is not None else 0
    ))

# Data sample
with st.expander("View Sample Data"):
    try:
        sample_data = pd.read_csv(DATA_FILE).head(5)
        st.dataframe(sample_data)
    except Exception as e:
        st.error(f"Could not load sample data: {str(e)}")
