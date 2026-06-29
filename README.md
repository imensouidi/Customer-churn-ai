# Customer Churn Prediction using Machine Learning & Deep Learning

## Project Overview

This project aims to predict customer churn for a subscription-based business using Machine Learning and Deep Learning techniques.

The objective is to identify customers who are likely to cancel their subscription in order to support customer retention strategies, improve customer retention and reduce revenue loss.

The project follows a complete Data Science workflow, from data exploration to the deployment of an interactive dashboard.

---

## Business Problem

Customer churn is a major challenge for subscription-based companies. Retaining existing customers is generally less expensive than acquiring new ones.

This project develops a predictive model capable of identifying customers at risk of churn, enabling businesses to take proactive retention actions.

The problem is formulated as a binary classification task:

- **0** → Customer stays
- **1** → Customer churns

---

## Project Structure

```text
Customer-Churn-AI/
│
├── data/
│   └── customer_churn_business_dataset.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── mlp.pkl
│   ├── preprocessor.pkl
│   ├── X_test_prepared.npy
│   └── y_test.npy
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses a synthetic business dataset containing **10,000 customers**.

The dataset includes demographic, behavioral, financial and customer satisfaction information.

### Main Features

### Customer Profile

- customer_id
- gender
- age
- country
- city
- customer_segment

### Subscription

- contract_type
- signup_channel
- tenure_months

### Customer Activity

- monthly_logins
- weekly_active_days
- avg_session_time
- features_used
- usage_growth_rate
- last_login_days_ago

### Financial Information

- monthly_fee
- total_revenue
- payment_method
- payment_failures
- discount_applied
- price_increase_last_3m

### Customer Support

- support_tickets
- avg_resolution_time
- complaint_type

### Customer Satisfaction

- csat_score
- nps_score
- survey_response

### Marketing

- email_open_rate
- marketing_click_rate
- referral_count

### Target Variable

- **0** → Customer stays
- **1** → Customer churns

---

## Project Notebooks

### 01 – Exploratory Data Analysis (EDA)

This notebook focuses on understanding the dataset through:

- Dataset overview
- Descriptive statistics
- Missing value analysis
- Duplicate detection
- Churn distribution
- Univariate and multivariate analyses
- Correlation analysis
- Data visualization

---

### 02 – Data Preprocessing

The preprocessing pipeline prepares the data for Machine Learning.

Main steps include:

- Duplicate removal
- Missing value verification
- Numerical and categorical feature separation
- Feature scaling
- One-Hot Encoding
- Train/Test split
- Random Under Sampling (RUS)

After preprocessing:

- **51 input features**
- **19 numerical variables**
- **32 categorical variables (One-Hot Encoding)**

---

### 03 – Modeling

Several predictive models were trained and compared.

Machine Learning models:

- Logistic Regression
- Random Forest
- XGBoost

Deep Learning model:

- Multi-Layer Perceptron (MLP)

Different imbalance handling strategies were evaluated.

The final selected model is **Random Forest combined with Random Under Sampling (RUS)**.

---

### 04 – Model Evaluation

The evaluation notebook validates the selected model using complementary evaluation techniques.

Performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Average Precision (PR-AUC)

Additional analyses:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Decision Threshold Optimization
- 5-Fold Cross Validation

Model interpretability:

- Feature Importance
- Permutation Importance
- SHAP Feature Importance
- SHAP Beeswarm
- SHAP Waterfall

These analyses provide both global and local explanations of the model predictions.

---

## Interactive Dashboard

An interactive Streamlit dashboard was developed to make the predictive model accessible to business users.

The application includes:

### Overview

- Business KPIs
- Churn distribution
- Customer segment analysis
- Contract type analysis
- Business impact estimation
- ROI estimation

### Customer Prediction

Users can simulate a customer profile and obtain:

- Churn probability
- Predicted class
- Risk level
- Risk factors
- CRM recommendations
- Estimated revenue at risk

### Model Performance

Comparison of the trained models through:

- Evaluation metrics
- ROC Curves
- Precision-Recall Curves
- Confusion Matrices

### Feature Analysis

Visualization of:

- Feature Importance
- Permutation Importance

to better understand the variables influencing churn prediction.

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- TensorFlow
- SHAP
- Streamlit
- Joblib
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-ai.git
```

Navigate to the project folder:

```bash
cd customer-churn-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the notebooks in the following order:

```text
01_EDA.ipynb
        ↓
02_Preprocessing.ipynb
        ↓
03_Modeling.ipynb
        ↓
04_Evaluation.ipynb
```

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Results

The comparative analysis showed that **Random Forest combined with Random Under Sampling (RUS)** achieved the best balance between predictive performance, robustness and interpretability.

The project demonstrates:

- End-to-end Data Science workflow
- Machine Learning and Deep Learning comparison
- Explainable Artificial Intelligence
- Business-oriented model evaluation
- Interactive decision-support dashboard

---

## Authors

- **Riad Boutria**
- **Imen Souidi**

---

## License

This project was developed for educational purposes.