# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:35:28 2025

@author: user
"""


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

st.title('Smart Churn App: AI-Powered Customer Retention')
st.text('AI-powered churn prediction platform for businesses. Analyze customer behavior, predict churn risk, and optimize retention strategies with data-driven insights.')

st.image('Title Image.jpg')
st.video('https://youtu.be/tnQVgzqAhQo?si=tQVeOdjuY02GCEeS')


# Apply a dynamic CSS style.
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    h1 {
        color: #ffcc00;
        text-align: center;
    }
    h2, h3, h4 {
        color: #ff6666;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#3a3a3a, #5a5a5a);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff5733;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #c70039;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header with a decorative banner
st.markdown("""
    <style>
        .rainbow-text {
            background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div style="text-align: center; background-color: #222; padding: 20px; border-radius: 10px; margin-bottom: 15px;">
        <h1 class="rainbow-text" style="font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 40px; font-weight: 600;">
            Streamlit Application for Customer Churn Analysis inside Telco
        </h1>
    </div>
    """, unsafe_allow_html=True)


# Loading of dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('Customer_Churn_Prediction.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_data(data):
    data = data.copy()
    # Drop customerID column if it exists
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
   
    # Replace inconsistent service strings
    data.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
   
    # Convert TotalCharges to numeric and fill missing values with the median
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
   
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x in ['Male', 'Yes'] else 0)
   
    # Function to encode three-option services
    def encode_service(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1  # Covers any non-standard response
   
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        data[col] = data[col].apply(encode_service)
   
    # Encode Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
    data['Contract'] = data['Contract'].apply(lambda x: 0 if x == "Month-to-month" else 1 if x == "One year" else 2)
   
    # Encode PaymentMethod: Electronic check -> 0, Mailed check -> 1, Bank transfer (automatic) -> 2, Credit card (automatic) -> 3
    data['PaymentMethod'] = data['PaymentMethod'].apply(
        lambda x: 0 if x == "Electronic check" else 1 if x == "Mailed check"
        else 2 if x == "Bank transfer (automatic)" else 3
    )
   
    # Ensure target variable (Churn) is numeric
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].apply(lambda x: 1 if x == "Yes" else 0)
   
    return data

df_processed = preprocess_data(df)

# Train-Test Split and Model Training
def train_model(data):
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = data[features]
    y = data['Churn']
   
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
   
    # Get feature importance
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
   
    return model, acc, cm, feature_importance, X_train, X_test, y_train, y_test

model, acc, cm, feature_importance, X_train, X_test, y_train, y_test = train_model(df_processed)

# Sidebar options for navigation
option = st.sidebar.selectbox(" Select your preferred Analysis",
                              [" Data Insights", " Interactive Charts", " Risk Assessment-Interactive Data Filtering", " Model Performance Analysis"])

# Page 1: Data Insights
if option == " Data Insights":
    st.image("Insights.jpeg")
    st.write("##  Data Insights")
    st.write(df.head())
    st.write("###  Data Summary")
    st.write(df.describe())

# Page 2: Interactive Charts

elif option == " Interactive Charts":
    # Section 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric("Total Customers", len(df))
   
    # Convert 'Churn' column from strings to numeric (1 for Yes, 0 for No)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
   
    with col2:
        churn_rate = df['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
       
    with col3:
        st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
       
    with col4:
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

    # Increase width of col1 (1:1 ratio)
    col1, col2 = st.columns([1, 1])  

    with col1:
        fig = px.pie(df, names='gender', title='Gender Distribution',
                     color_discrete_sequence=['#B8860B', '#8B0000', '#006400'])  # Dark Yellow, Dark Red, Dark Green
   
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(width=400, height=500)
   
        st.plotly_chart(fig, use_container_width=True)
       
    with col2:
        fig = px.histogram(df, x='tenure', nbins=20, title='Tenure Distribution',
                           color_discrete_sequence=['#8B0000'])  # Dark Red
        fig.update_layout(bargap=0.1)
        fig.update_layout(width=1000, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df, x='Contract', y='Churn', color='Contract',
                     title='Churn Rate by Contract Type',
                     color_discrete_sequence=['#8B0000', '#32CD32', '#FFD700'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                         title='Monthly Charges vs Tenure',
                         color_discrete_sequence=['#8B0000', '#32CD32'])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names='PaymentMethod', title='Payment Method Distribution',
                     color_discrete_sequence=['#DC143C', '#008000', '#00008B', '#FFD700'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.header(" Service Usage Patterns")
    service_cols = ['PhoneService', 'InternetService', 'StreamingTV', 'TechSupport']
    selected_service = st.selectbox("Select Service", service_cols)

    fig = px.bar(df, x=selected_service, color='Churn', barmode='group',
                 title=f'{selected_service} vs Churn',
                 color_discrete_sequence=['#B8860B', '#006400'])  # Dark Yellow, Dark Green

    st.plotly_chart(fig, use_container_width=True)


# Page 3: Churn Prediction Model with Risk Assessment and Interactive Data Filtering

elif option == " Risk Assessment-Interactive Data Filtering":
    st.image("Churn.png")
    st.write("## Churn Prediction Model with Risk Assessment and Interactive Data Filtering")

    # Add model selection dropdown
    model_choice = st.selectbox("Select Prediction Model", ["Random Forest", "K-Means Clustering", "Logistic Regression"])

    # User input for features
    tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)
   
    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
   
    # Encode inputs consistently with training
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_service_encoded = 1 if phone_service == "Yes" else 0
   
    def encode_service_input(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1
   
    multiple_lines_encoded = encode_service_input(multiple_lines)
    online_security_encoded = encode_service_input(online_security)
    online_backup_encoded = encode_service_input(online_backup)
    device_protection_encoded = encode_service_input(device_protection)
    tech_support_encoded = encode_service_input(tech_support)
    streaming_tv_encoded = encode_service_input(streaming_tv)
    streaming_movies_encoded = encode_service_input(streaming_movies)
   
    contract_encoded = 0 if contract=="Month-to-month" else 1 if contract=="One year" else 2
    paperless_billing_encoded = 1 if paperless_billing=="Yes" else 0
    payment_method_encoded = (0 if payment_method=="Electronic check"
                              else 1 if payment_method=="Mailed check"
                              else 2 if payment_method=="Bank transfer (automatic)" else 3)
   
    # Create input data DataFrame
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender_encoded],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'PhoneService': [phone_service_encoded],
        'MultipleLines': [multiple_lines_encoded],
        'OnlineSecurity': [online_security_encoded],
        'OnlineBackup': [online_backup_encoded],
        'DeviceProtection': [device_protection_encoded],
        'TechSupport': [tech_support_encoded],
        'StreamingTV': [streaming_tv_encoded],
        'StreamingMovies': [streaming_movies_encoded],
        'Contract': [contract_encoded],
        'PaperlessBilling': [paperless_billing_encoded],
        'PaymentMethod': [payment_method_encoded]
    })

    # Train Logistic Regression Model
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    # Train K-Means Model (using 2 clusters: Churn & No Churn)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    if st.button("Predict Churn"):
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)
        elif model_choice == "Logistic Regression":
            prediction = log_reg_model.predict(input_data)
        elif model_choice == "K-Means Clustering":
            prediction = kmeans.predict(input_data)

        if prediction[0] == 1:
            st.write("**Prediction: This customer is likely to churn.**")
        else:
            st.write("**Prediction: This customer is likely to stay.**")

# Page 4: Model Performance Analysis

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Train K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)
kmeans_pred = kmeans.predict(X_test)

# Train Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_test)


# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
kmeans_acc = accuracy_score(y_test, kmeans_pred)
log_reg_acc = accuracy_score(y_test, log_reg_pred)

# Confusion Matrices
rf_cm = confusion_matrix(y_test, rf_pred)
kmeans_cm = confusion_matrix(y_test, kmeans_pred)
log_reg_cm = confusion_matrix(y_test, log_reg_pred)

# Feature Importance (Only for models that support it)
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)

# Logistic Regression Feature Importance (absolute coefficients)
log_reg_feature_importance = pd.Series(abs(log_reg_model.coef_[0]), index=X_train.columns)

# K-Means Feature Importance (not available, so using cluster centers)
kmeans_feature_importance = pd.Series(kmeans.cluster_centers_[0], index=X_train.columns)

if option == " Model Performance Analysis":
    st.image("Performance.png")
    st.write("## Model Evaluation Metrics")

    selected_model = st.selectbox("Select Model", ["Random Forest", "K-Means Clustering", "Logistic Regression"])

    if selected_model == "Random Forest":
        st.write(f"**Model Accuracy:** {rf_acc:.2f}")
        cm, feature_importance = rf_cm, rf_feature_importance
        cmap = "Blues"
        feature_color = "cyan"
    elif selected_model == "K-Means Clustering":
        st.write(f"**Model Accuracy:** {kmeans_acc:.2f}")
        cm, feature_importance = kmeans_cm, kmeans_feature_importance
        cmap = "Greens"
        feature_color = "limegreen"
    elif selected_model == "Logistic Regression":
        st.write(f"**Model Accuracy:** {log_reg_acc:.2f}")
        cm, feature_importance = log_reg_cm, log_reg_feature_importance
        cmap = "Purples"
        feature_color = "purple"

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    st.pyplot(fig)

    # Feature Importance
    st.write("### Feature Importance")
    fig, ax = plt.subplots()
    feature_importance.plot(kind='bar', ax=ax, color=feature_color)
    st.pyplot(fig)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    numeric_df = df_processed.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

