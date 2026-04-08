import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Title
st.title("🏢 Employee Attrition Prediction System")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Data Exploration", "Model Performance", "Make Prediction"]
)

# Load data
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "Employee Attrition Data (1).csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    return pd.read_csv(data_path)

# Preprocess data
@st.cache_data
def preprocess_data(df):
    # Drop null values
    df = df.dropna()
    
    # Filter numeric values
    df = df[df['Age'].str.isnumeric()]
    df = df[df['JobSatisfaction'].str.isnumeric()]
    
    # Convert to int
    df['Age'] = df['Age'].astype(int)
    df['JobSatisfaction'] = df['JobSatisfaction'].astype(int)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Replace Yes/No values
    df["OverTime"] = df["OverTime"].replace({"No":0,"Yes":1,"N":0,"Y":1,"no":0,"yes":1}).astype(int)
    df["Attrition"] = df["Attrition"].replace({"No":0,"Yes":1,"N":0,"Y":1,"no":0,"yes":1}).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Setup encoder
    ohe = OneHotEncoder()
    ohe.fit(df[["Department"]])
    
    # Create column transformer
    ct = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ["Department"]), 
        remainder="passthrough"
    )
    
    # Prepare X and y
    x = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Create and train pipeline
    lr = LogisticRegression()
    pipe = make_pipeline(ct, lr)
    pipe.fit(x_train, y_train)
    
    # Save model
    joblib.dump(pipe, 'logistic.pkl')
    
    # Predictions
    y_pred = pipe.predict(x_test)
    
    return pipe, x_train, x_test, y_train, y_test, y_pred, x, y

# Load preprocessed data
df = load_data()
df = preprocess_data(df)

# Train model
pipe, x_train, x_test, y_train, y_test, y_pred, x, y = train_model(df)

# Home Page
if page == "Home":
    st.header("Welcome to Employee Attrition Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        attrition_rate = df['Attrition'].sum() / len(df) * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Features of this Application:
    
    - **Data Exploration**: View and analyze employee data
    - **Model Performance**: Check model metrics and visualization
    - **Make Predictions**: Predict employee attrition for new employees
    
    ### 🔍 Features Used:
    - Age
    - Job Satisfaction
    - Department
    - OverTime status
    - And other employee attributes
    """)

# Data Exploration Page
elif page == "Data Exploration":
    st.header("📊 Data Exploration")
    
    # Display dataset info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display first few rows
    st.subheader("First Few Records")
    st.dataframe(df.head(10))
    
    # Dataset info
    st.subheader("Dataset Info")
    st.write(df.info())
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Attrition distribution
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    attrition_counts = df['Attrition'].value_counts()
    ax.bar(['No Attrition', 'Attrition'], attrition_counts.values, color=['green', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Employee Attrition Distribution')
    st.pyplot(fig)
    
    # Department distribution
    st.subheader("Department Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    dept_counts = df['Department'].value_counts()
    ax.bar(dept_counts.index, dept_counts.values, color='skyblue')
    ax.set_ylabel('Count')
    ax.set_title('Employees by Department')
    st.pyplot(fig)
    
    # Age distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['Age'], bins=20, color='purple', edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Age Distribution of Employees')
    st.pyplot(fig)

# Model Performance Page
elif page == "Model Performance":
    st.header("📈 Model Performance")
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy Score", f"{accuracy:.4f}")
    with col2:
        st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Model Details
    st.subheader("Model Details")
    st.write("""
    **Model Type:** Logistic Regression
    **Training Set Size:** {} samples
    **Test Set Size:** {} samples
    **Total Features:** {}
    """.format(len(x_train), len(x_test), x.shape[1]))

# Prediction Page
elif page == "Make Prediction":
    st.header("🔮 Make Prediction")
    
    st.write("Enter employee details to predict attrition probability:")
    
    # Get unique values for selectors
    departments = sorted(df['Department'].unique())
    
    # Create input columns
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=int(df['Age'].min()), 
                       max_value=int(df['Age'].max()), 
                       value=int(df['Age'].mean()))
        
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 
                                     min_value=1, max_value=4, value=3)
    
    with col2:
        department = st.selectbox("Department", departments)
        
        overtime = st.selectbox("Over Time", ["No", "Yes"])
    
    # Make prediction
    if st.button("Predict Attrition", key="predict_btn"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'JobSatisfaction': [job_satisfaction],
            'Department': [department],
            'OverTime': [1 if overtime == "Yes" else 0]
        })
        
        # Add other required columns with default values
        for col in x.columns:
            if col not in input_data.columns:
                input_data[col] = df[col].iloc[0]
        
        # Reorder columns to match training data
        input_data = input_data[x.columns]
        
        # Make prediction
        prediction = pipe.predict(input_data)[0]
        probability = pipe.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ **Not Likely to Leave**")
                st.write(f"Probability of staying: {probability[0]:.2%}")
            else:
                st.error("⚠️ **Likely to Leave**")
                st.write(f"Probability of leaving: {probability[1]:.2%}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Stay', 'Leave']
            sizes = [probability[0] * 100, probability[1] * 100]
            colors = ['green', 'red']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Attrition Probability')
            st.pyplot(fig)
        
        # Employee profile
        st.subheader("Employee Profile Summary")
        profile_data = {
            'Attribute': ['Age', 'Job Satisfaction', 'Department', 'Over Time'],
            'Value': [age, job_satisfaction, department, overtime]
        }
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit | Employee Attrition Prediction")
