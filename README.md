# RetainFlow Automation: Customer Churn Prediction

## Project Overview
This project is a system that automates customer retention strategies through customer churn prediction and revenue prediction. Real-time prediction and data storage are possible through a web application using Streamlit and Supabase database integration.

## Key Features
- **Customer Churn Prediction**: Customer churn probability prediction through machine learning models
- **Revenue Prediction**: Customer expected revenue prediction
- **Customer Segmentation**: Customer classification by risk and value
- **Supabase Integration**: Real-time data storage and management
- **Streamlit Cloud Deployment**: Web-based interface provision
- **Automation Workflow**: Task and data pipeline automation powered by n8n & Make 

## Models Used

### 1. Customer Churn Prediction Model
- **Problem Type**: Classification
- **Model**: RandomForestClassifier + CalibratedClassifierCV
- **Purpose**: Predict customer churn probability (0-1)
- **Features**: Customer tenure, contract type, payment method, service usage, billing data
- **Preprocessing**: SMOTE oversampling for class imbalance, one-hot encoding
- **Segmentation**: KMeans clustering (4 groups based on churn probability and monthly charges)

### 2. Revenue Prediction Model
- **Problem Type**: Regression
- **Model**: Baseline Linear Regression + RandomForestRegressor (Residual Learning)
- **Purpose**: Predict customer total charges/revenue
- **Features**: 
  - Baseline: tenure, monthly charges
  - Residual: payment method, service features, demographics
- **Methodology**: Residual learning to correct baseline prediction errors

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python, SQLAlchemy
- **Database**: Supabase (PostgreSQL)
- **ML**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Automation Tool**: n8n & Make

### Workflow Details

#### Trigger
- When new prediction data (e.g., **churn probability > threshold**) is inserted into **Supabase**.

#### n8n Role
- Monitors **Supabase** via Webhook or Supabase node  
- Filters customers by **churn probability** or **revenue tier**  
- Sends structured **JSON payload** to **Make** or other external APIs

#### Make Role
- Executes automated actions such as:
  - Sending notifications (e.g., Slack, Gmail)
  - Updating CRM tools (**HubSpot**, **Notion**, etc.)
  - Creating follow-up tasks for the retention team

#### Example Use Case
> A customer with **churn probability > 0.8** is detected →  
> **n8n** triggers a webhook →  
> **Make** sends a **Slack alert** to the retention team and creates a **follow-up task** in Notion.

---

### 🧩 Benefits
- **Fully automated** data-driven retention workflow  
- **No manual** data checking or script execution required  
- **Scalable integration** with 3rd-party tools (Slack, Gmail, Notion, HubSpot, etc.)


## Streamlit Cloud Deployment Guide

### 1. Prepare GitHub Repository
1. Upload this project to a GitHub repository
2. Verify all files are in the correct location

### 2. Streamlit Cloud Setup
1. Access [Streamlit Cloud](https://share.streamlit.io/)
2. Click "New app"
3. Connect GitHub repository
4. Main file path: `Streamlit_app.py`

### 3. Environment Variables Setup
Set the following environment variables in Streamlit Cloud's Secrets Management:

```toml
[DATABASE_URL]
DATABASE_URL = "postgresql+psycopg2://postgres:YOUR_PASSWORD@db.fjaxvaegmtbsyogavuzy.supabase.co:5432/postgres?sslmode=require"
```

### 4. Required File Structure
```
├── Streamlit_app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── notebook/
│   ├── pipeline_customer_churn_model.pkl
│   ├── pipeline_customer_revenue_model.pkl
│   ├── notebook/
│   │   ├── eda_insight/
│   │   └── modeling_insight/
│   └── revenue_insight/
└── pic.png
```

## Local Execution
```bash
pip install -r requirements.txt
streamlit run Streamlit_app.py
```

## Troubleshooting

### 1. Model File Not Found Error
- Check if model files (`pipeline_customer_churn_model.pkl`, `pipeline_customer_revenue_model.pkl`) are in the `notebook/` directory
- Verify all files are uploaded to GitHub

### 2. Supabase Connection Failure
- Set correct DATABASE_URL in Streamlit Cloud's Secrets Management
- Verify Supabase project is active
- Check firewall settings

### 3. psycopg2-binary Installation Failure
- Set psycopg2-binary version to 2.9.10 or higher in requirements.txt
- Streamlit Cloud automatically installs compatible versions

## Notes
- Supabase database connection information is managed through environment variables
- Model files and image files are referenced with relative paths
- Absolute paths cannot be used in Streamlit Cloud
- Prediction functionality works normally even if DB connection fails
