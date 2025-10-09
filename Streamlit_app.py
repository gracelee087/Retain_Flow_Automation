import pandas as pd
import streamlit as st
import pickle
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.postgresql import insert
import numpy as np
import os

# Supabase DB connection string - using environment variables
# DB connection is not tested at app startup (only connected in Application tab)


    
st.title("RetainFlow Automation: Customer Churn Prediction")


# Create tabs
tab1, tab2, tab3, tab4, tab5= st.tabs(["Problem", "EDA", "Modeling/Evaluation", "Application", "Outcome"])



with tab1:

    # Objective
    st.header("Objective")
    st.markdown("""
    - Identify churn-prone customers using historical telco data  
    - Automate retention workflows by connecting predictions to **make.com**  
    - Enable timely customer engagement (e.g., sending offers, alerts to CRM, or triggering support tickets)  
    """)


    st.header("Business Problem")
    st.markdown("""
    - Telecom customers can easily switch, so **churn directly reduces revenue**.  
    - Even a small improvement matters: a **5% reduction in churn can boost profits by 25–95%** (Bain & Co.).  
    - **Customer Acquisition Cost (CAC)** is about **5x higher** than **Customer Retention Cost (CRC)** (HBR).  
    - Despite this, many churn analyses stop at *prediction*, without seamless integration into **real business workflows**.  
    - Marketing and support teams often lack **real-time triggers** to act on churn insights.  
    - Without automation, valuable time is lost between churn detection and customer outreach.  
    - **Key challenge**: How well churn is predicted **and** how effectively marketing actions are automated determines revenue growth.  
    - This project bridges the gap by combining **machine learning models** with **make.com automation**, enabling organizations to not only *predict churn* but also to **automatically take retention actions**.  
    """)



    st.header("Telco Opportunity Map")
    st.image("pic.png", caption="Opportunity Map: Balancing Effort vs. Benefit", width=700)

    st.markdown("""
This opportunity map helps us **prioritize projects**:  
- **Top-left (Quick Wins)**: High benefit, low effort → e.g., Churn AI + Discounts, Loyalty programs.  
- **Bottom-right (Long-term Bets)**: High effort, uncertain benefit → e.g., IoT, 5G marketing.  
- Our focus starts with **Churn AI**, where benefit is high and execution is feasible.  
    """)



    st.header("Why It Matters")
    st.markdown("""
- **Cost efficiency**: Retaining is cheaper & more profitable.  
- **Customer lifetime value**: Identify & prioritize high-value churn-risk customers.  
- **Personalized marketing**: Segmentation + churn probability → tailored offers.  
- **Revenue impact**: Proactive churn management drives growth.  
- **Strategic decisions**: Data-driven CRM, bundles, loyalty programs.  
    """)




    # 참고문헌 (작게 표시)
    st.markdown(
        """
        <sub>**References**:  
        Bain & Company, *Customer Retention Economics*  
        Harvard Business Review (2014), *The Value of Keeping the Right Customers*</sub>
        """,
        unsafe_allow_html=True
    )














with tab2:


    
    st.header("Exploratory Data Analysis Results")




    st.markdown("""
                

    ### Data source / collection / challenges

    **Data source**  
    - Telco Customer Churn dataset (Kaggle / IBM Sample Data)  
    - Includes customer contracts, payment methods, service usage, billing history, and churn labels  

    **Data collection**  
    - Key feature categories:  
    - **Customer info**: tenure, SeniorCitizen  
    - **Contract info**: Contract, PaymentMethod, InternetService  
    - **Service usage**: TechSupport, OnlineSecurity, StreamingTV  
    - **Billing data**: MonthlyCharges, TotalCharges  
    - Churn is defined as whether a customer discontinued the service within a given period  

    **Challenges**  
    - **Class imbalance**: majority of customers are `Churn=No`, while `Churn=Yes` is a minority → risk of biased models  
    - **Categorical features**: contract, payment method, and service usage require encoding for ML models  
    - **Limitations in realism**:  
    - Lacks behavioral data such as complaints, service quality issues, or customer interactions  
    - No information on customer re-subscription after churn or the impact of specific marketing campaigns  
    - Therefore, assessing marketing effectiveness and retention strategies is challenging with this dataset

    ---

                
    ### Key Insights from EDA

    **Customer Tenure**  
    - Customers with shorter tenure show a higher churn rate  
    - Long-term customers tend to have higher TotalCharges and lower churn probability  

    **Billing Metrics**  
    - Higher MonthlyCharges are associated with higher churn  
    - In contrast, TotalCharges show a negative correlation with churn, reflecting stronger customer loyalty 
      (Higher total charges (cumulative spending) indicate lower churn probability) 

    **Contract & Service Features**  
    - Month-to-month contracts have the highest churn rate  
    - Customers paying via **Electronic check** are more likely to churn  
    - Customers without **TechSupport / OnlineSecurity** services show significantly higher churn  

    """)











    eda_path = "notebook/notebook/eda_insight"

    if os.path.exists(eda_path):
        img_files = [f for f in os.listdir(eda_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if img_files:
            for img in img_files:
                st.image(
                    os.path.join(eda_path, img),
                    caption=img
                )
        else:
            st.warning("⚠️ EDA image files not found.")
    else:
        st.error("❌ EDA path not found. Please check the path again.")



with tab3:


    # ---------------------------
    # Methodology & Technology Stack
    # ---------------------------
    st.header("Methodology and Technology Stack Used")

    st.subheader("1. Methodology")
    st.markdown("""
    **1) Customer Churn Prediction**  
    - Data preprocessing: missing value imputation, numeric conversion, one-hot encoding  
    - Handling class imbalance: **SMOTE oversampling**  
    - Model: **RandomForestClassifier + CalibratedClassifierCV** for probability calibration  
    - Customer segmentation: **KMeans clustering (4 groups based on churn probability and Monthly Charges)**  
        ChurnProbability(High risk/Low risk), 
        MonthlyCharges(high value/low value)

    **2) Revenue / Total Charges Prediction**  
    - Baseline model + Residual model (**Residual Learning**)  
    - Residual **RandomForestRegressor** to correct baseline prediction errors  
    """)

    st.subheader("Technology Stack")
    st.markdown("""
    - **Python 3.10+**  
    - **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib, cloudpickle  
    - **Deployment / Visualization**: Streamlit  
    - **Environment**: Jupyter Notebook, Virtual Environment (venv)  
    """)


    # ---------------------------
    # Modelling
    # ---------------------------
    st.header("2. Modeling")

    st.markdown("""
    **1) Customer Churn Prediction Model** 
    - **Problem Type**: Classification  
    - **Model Used**: RandomForestClassifier + CalibratedClassifierCV + KMeans  
    - **Output**: Churn probability (0–1)  
    - **Evaluation Metrics**: Accuracy, Recall, Precision, F1, ROC-AUC  
    - **Calibration**: Sigmoid-based probability calibration  
    - **Loss Function / Activation Function**:  
        - Loss: None explicitly (tree-based split criteria = Gini / Cross-Entropy)  
        - Activation: None (Sigmoid in CalibratedClassifierCV for probability conversion)  
    - **Notes**: KMeans segmentation identifies 4 customer groups for additional insights  
    """)

    st.markdown("""
    **2) Revenue / Total Charges Prediction Model** 
    - **Problem Type**: Regression  
    - **Model Used**: Baseline model + RandomForest residual model  
    - **Output**: Continuous value (Total Charges)  
    - **Evaluation Metrics**: RMSE, R²  
    - **Loss Function / Activation Function**:  
        - Loss: MSE (variance minimization in RandomForest regressor)  
        - Activation: None  
    - **Notes**: Residual learning corrects baseline model errors to improve prediction accuracy  
    """)








    st.header("Churn Prediction Model Performance Evaluation")

    modeling_path = "notebook/notebook/modeling_insight"

    st.markdown("""
    - Before calibration, the model achieved Recall = 0.614, ROC AUC = 0.834.
    - After calibration, the results were Recall = 0.529, ROC AUC = 0.835. 
        """)


    if os.path.exists(modeling_path):
        img_files = [f for f in os.listdir(modeling_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if img_files:
            for img in sorted(img_files):  # Sort to display in order
                st.image(
                    os.path.join(modeling_path, img),
                    caption=img,
                    width=800 # ✅ Desired size (px unit)
                )
        else:
            st.warning("⚠️ Model performance image files not found.")
    else:
        st.error("❌ Modeling results path not found. Please check the path again.")












    # ---------------------------
    # Separator between Churn model and Revenue model
    # ---------------------------
    

    st.divider()  # Latest Streamlit
    # st.markdown("---")  # Or this method is also possible

    st.header("Revenue Prediction Model Performance Evaluation")

    revenue_path = "notebook/revenue_insight"

    st.markdown("### Baseline Revenue Model (Linear Regression)")
    st.write("- Baseline R²: 0.8902 (≈ 0.89), TotalCharges")
    st.write("- Residual R²: 0.5444 (≈ 0.55)) → Indicates that some complex patterns are not explained by the baseline model")

    st.markdown("### Residual Model (RandomForestRegressor)")
    st.write("- Target: Residuals")

    st.markdown("### Combined Model (Baseline + Residual)")
    st.write("- Final R² ≈ 0.965 (96.5%)")
    st.write("- Final RMSE ≈ 423.9 (18%)")

    if os.path.exists(revenue_path):
        img_files = [f for f in os.listdir(revenue_path) if f.endswith(".png")]
        if img_files:
            for img in sorted(img_files):
                st.image(
                    os.path.join(revenue_path, img),
                    caption=img,
                    width=800
                )
        else:
            st.warning("⚠️ Revenue model visualization images not found.")
    else:
        st.error("❌ Revenue results path not found. Please check the path again.")







with tab4:
    st.header("Customer Churn + Revenue Forecasting (Supabase Integration)")

    # ---------------------------
    # 1. Load models (only when user clicks the tab)
    # ---------------------------
    @st.cache_resource
    def load_models():
        try:
            import warnings
            import sys
            warnings.filterwarnings("ignore")
            
            # Patch to bypass scikit-learn compatibility issues
            try:
                from sklearn.utils._tags import _safe_tags
            except ImportError:
                # Create dummy function if _safe_tags is not available
                def _safe_tags(estimator, key, tag_val=None):
                    return getattr(estimator, '_more_tags', lambda: {}).get(key, tag_val)
                
                import sklearn.utils._tags
                sklearn.utils._tags._safe_tags = _safe_tags
            
            with open("notebook/pipeline_customer_churn_model.pkl", "rb") as f:
                bundle = pickle.load(f)

            model = bundle["model"]
            scaler = bundle["scaler"]
            kmeans = bundle["kmeans"]

            with open("notebook/pipeline_customer_revenue_model.pkl", "rb") as f:
                revenue_bundle = pickle.load(f)

            base_model = revenue_bundle["baseline_model"]
            residual_model = revenue_bundle["residual_model"]
            
            return model, scaler, kmeans, base_model, residual_model, None
            
        except FileNotFoundError as e:
            return None, None, None, None, None, f"❌ Model file not found: {e}"
        except Exception as e:
            return None, None, None, None, None, f"❌ Model loading failed: {e}"

    # 모델 로드 시도
    model, scaler, kmeans, base_model, residual_model, error = load_models()
    
    if error:
        st.error(error)
        st.info("💡 Please check if the model files are in the correct location.")
        st.stop()
    else:
        st.success("✅ Model loaded successfully")

    # ---------------------------
    # 2. Postgres DB connection
    # ---------------------------
    try:
        # Get DATABASE_URL correctly
        if isinstance(st.secrets["DATABASE_URL"], dict):
            database_url = st.secrets["DATABASE_URL"]["DATABASE_URL"]
        else:
            database_url = st.secrets["DATABASE_URL"]
            
        engine = create_engine(database_url)
        # Connection test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.success("✅ Supabase DB connected successfully")
        db_connected = True
    except Exception as e:
        st.warning(f"⚠️ Supabase DB connection failed: {e}")
        st.info("💡 Prediction functionality can be used without DB connection.")
        st.info("💡 Set DATABASE_URL in Streamlit Cloud's Secrets Management.")
        db_connected = False

    # ---------------------------
    # 4. Streamlit UI
    # ---------------------------
    uploaded_file = st.file_uploader("Customer CSV upload", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # (1) Customer churn probability prediction
        df["churn_prob"] = model.predict_proba(df)[:, 1]

        # (2) Revenue prediction (Baseline + Residual)
        X_base = df[["tenure", "MonthlyCharges"]]
        X_res = df[["PaymentMethod", "PaperlessBilling", "Dependents",
                    "OnlineBackup", "InternetService", "StreamingTV", "OnlineSecurity"]]

        baseline_pred = base_model.predict(X_base)
        residual_pred = residual_model.predict(X_res)
        df["predicted_revenue"] = np.clip(baseline_pred + residual_pred, a_min=0, a_max=None)

        # (3) 📌 12-month based metrics
        df["revenue_12m"] = df["MonthlyCharges"] * 12
        df["expected_loss_12m"] = df["revenue_12m"] * df["churn_prob"]

        # (4) Clustering
        cluster_input = pd.DataFrame({
            "ChurnProbability": df["churn_prob"],
            "MonthlyCharges": df["MonthlyCharges"]
        })
        df["Cluster"] = kmeans.predict(scaler.transform(cluster_input))

        # 📊 Automatic Risk/Value labeling based on cluster averages
        cluster_summary = df.groupby("Cluster")[["churn_prob", "MonthlyCharges"]].mean()
        risk_threshold = df["churn_prob"].mean()
        value_threshold = df["MonthlyCharges"].mean()

        def auto_label(cluster):
            row = cluster_summary.loc[cluster]
            risk = "High Risk" if row["churn_prob"] >= risk_threshold else "Low Risk"
            value = "High Value" if row["MonthlyCharges"] >= value_threshold else "Low Value"
            return f"{risk} & {value}"

        df["cluster_label"] = df["Cluster"].apply(auto_label)

        # (5) Generate base_message
        base_messages = {
            "High Risk & High Value": "Exclusive premium offers to retain our top customers",
            "High Risk & Low Value": "Special discount to prevent churn at minimal cost",
            "Low Risk & High Value": "VIP thank-you campaign for loyal high-value customers",
            "Low Risk & Low Value": "Customer feedback request to strengthen relationships",
        }
        df["base_message"] = df["cluster_label"].map(base_messages)


        # (6) Match column names with DB table
        df = df.rename(columns={
            "customerID": "customer_id",
            "Email": "email"
        })


        # ---------------------------
        # 7. Supabase DB save (only when DB is connected)
        # ---------------------------
        if db_connected:
            try:
                metadata = MetaData()
                metadata.reflect(bind=engine)
                predictions_table = metadata.tables["predictions"]

                with engine.begin() as conn:
                    for _, row in df.iterrows():
                        stmt = insert(predictions_table).values(
                            customer_id=row["customer_id"],
                            email=row["email"],
                            churn_prob=row["churn_prob"],
                            cluster_label=row["cluster_label"],
                            base_message=row["base_message"],
                            predicted_revenue=row["predicted_revenue"],
                            revenue_12m=row["revenue_12m"],
                            expected_loss_12m=row["expected_loss_12m"]
                        )
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["customer_id"],
                            set_={
                                "email": row["email"],
                                "churn_prob": row["churn_prob"],
                                "cluster_label": row["cluster_label"],
                                "base_message": row["base_message"],
                                "predicted_revenue": row["predicted_revenue"],
                                "revenue_12m": row["revenue_12m"],
                                "expected_loss_12m": row["expected_loss_12m"]
                            }
                        )
                        conn.execute(stmt)

                # ---------------------------
                # 8. Save Top 10 customers (→ top_risk_customers table)
                # ---------------------------
                top10 = df.sort_values("expected_loss_12m", ascending=False).head(10)

                top_table = metadata.tables["top_risk_customers"]

                with engine.begin() as conn:
                    # Clear existing data and save new data (overwrite method)
                    conn.execute(text("TRUNCATE TABLE top_risk_customers;"))

                    for _, row in top10.iterrows():
                        stmt = insert(top_table).values(
                            customer_id=row["customer_id"],
                            email=row["email"],
                            churn_prob=row["churn_prob"],
                            cluster_label=row["cluster_label"],
                            base_message=row["base_message"],
                            predicted_revenue=row["predicted_revenue"],
                            revenue_12m=row["revenue_12m"],
                            expected_loss_12m=row["expected_loss_12m"]
                        )
                        conn.execute(stmt)
                        
                st.success("✅ Supabase DB update completed! (All predictions + Top 10 saved)")
                
            except Exception as e:
                st.error(f"❌ DB save failed: {e}")
                st.info("💡 Prediction results are displayed but not saved to DB.")
        else:
            st.info("ℹ️ Data not saved due to no DB connection.")

        # ---------------------------
        # 9. Streamlit output
        # ---------------------------
        st.subheader("Prediction and segmentation results")
        st.dataframe(df[["customer_id", "email", "churn_prob",
                        "cluster_label", "base_message",
                        "predicted_revenue",
                        "revenue_12m", "expected_loss_12m"]])

        st.subheader("Top 10 Revenue at Risk Customers (12-Month Basis)")
        st.dataframe(top10)




with tab5:

    st.header("Next Steps & Open Challenges")

    st.markdown("""
- **Data expansion**: Include call center logs, customer support chats, and usage data.  
- **Real-time integration**: Connect models directly with CRM for live churn alerts.  
- **A/B testing**: Validate the effectiveness of personalized retention campaigns.  
- **Churn model optimization**: Explore XGBoost/LightGBM, better calibration, and automated threshold tuning.  
- **Model robustness**: How well does the model generalize across new customer cohorts?  
- **Ethics & fairness**: Could targeting strategies unintentionally bias or exclude groups?  
    """)

    st.divider()

    st.header("Long-term Vision")
    st.markdown("""
- Build **automation pipelines** with high-performing models beyond churn/revenue.  
- Enable **1 person to deliver the productivity of 10** through intelligent automation.  
- Move toward a future where **data-driven decision-making** is seamlessly embedded in daily operations.  
    """)
















