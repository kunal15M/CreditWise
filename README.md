# 🏦 CreditWise: Loan Approval Predictor

CreditWise is an end-to-end Machine Learning project that predicts whether a loan application will be approved or rejected based on applicant information. Financial institutions rely on automation and data-driven insights to evaluate loan eligibility, and this project serves as a scalable solution to automate that process.

## 🌟 Project Overview

The project follows a complete machine learning workflow:
- **Data Preprocessing:** Handled missing values (mean/mode imputation), performed feature scaling, and applied one-hot encoding to categorical variables.
- **Exploratory Data Analysis (EDA):** Visualized distributions and relationships using Seaborn and Matplotlib to understand key drivers for loan approval.
- **Modeling:** Built and evaluated three classification algorithms:
  - Logistic Regression (Selected for the final pipeline)
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
- **Productionization:** Bundled the final model and preprocessing steps into an `sklearn` `Pipeline` and serialized it using `joblib`.
- **Web Interface:** A beautiful, responsive dashboard built with Streamlit allows users to input applicant details and receive real-time predictions.

## 📂 Project Structure

```text
First_Minor_project/
├── loan_approval_data.csv          # Raw dataset containing applicant information
├── loan_approval_predicter.ipynb   # Jupyter Notebook with EDA, training, and evaluation
├── train.py                        # Script to re-train and export the model pipeline
├── creditwise_pipeline.joblib      # Serialized scikit-learn Pipeline (Model + Preprocessor)
├── app.py                          # Streamlit frontend for the prediction app
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed, along with the required dependencies.
You can install the necessary packages using pip:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib streamlit joblib
```

### 1. Training the Model
If you want to train the model from scratch and generate the `creditwise_pipeline.joblib` file, run:
```bash
python train.py
```
*(Alternatively, you can run the cells in `loan_approval_predicter.ipynb`)*

### 2. Running the Web App
To start the Streamlit web application and interact with the model:
```bash
streamlit run app.py
```
This will launch a local server and open the web dashboard in your default browser.

## 📊 Dataset Features

The model evaluates a variety of demographic and financial indicators:
- **Financials:** Applicant Income, Coapplicant Income, Savings, Collateral Value
- **Loan Details:** Loan Amount, Loan Term, Loan Purpose, Existing Loans
- **Credit Health:** Credit Score, Debt-to-Income (DTI) Ratio
- **Demographics:** Age, Gender, Marital Status, Dependents, Education Level
- **Employment:** Employment Status, Employer Category

## 🛠️ Built With

- **Python:** Primary programming language
- **Scikit-Learn:** Machine Learning models and preprocessing pipelines
- **Pandas & NumPy:** Data manipulation and analysis
- **Streamlit:** Interactive web application framework
- **Matplotlib & Seaborn:** Data visualization
