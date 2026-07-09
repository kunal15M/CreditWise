# CreditWise: Loan Approval Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a loan application is likely to be approved based on applicant and financial information. This project demonstrates the complete machine learning workflow, from data preprocessing and exploratory data analysis to model training, evaluation, and feature engineering.

---

## Project Overview

Financial institutions receive thousands of loan applications, making manual approval processes time-consuming and prone to inconsistencies. This project uses supervised machine learning algorithms to automate loan approval prediction based on historical applicant data.

The notebook follows a structured machine learning pipeline and compares multiple classification algorithms to determine the most effective model.

---

## Dataset

The dataset contains applicant information such as:

- Applicant Income
- Co-applicant Income
- Loan Amount
- Credit Score
- Debt-to-Income Ratio
- Employment Status
- Education Level
- Marital Status
- Property Area
- Loan Term
- Loan Approval Status (Target Variable)

---

## Project Workflow

### Data Exploration
- Loaded and inspected the dataset
- Examined data types and summary statistics
- Identified missing values and duplicates

### Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Scaled numerical features
- Split the dataset into training and testing sets

### Exploratory Data Analysis (EDA)
- Distribution plots
- Count plots
- Pie charts
- Box plots
- Correlation heatmap

### Machine Learning Models
The following classification algorithms were implemented:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

### Feature Engineering
Additional features were created to improve predictive performance, including polynomial transformations of selected numerical variables.

### Model Evaluation
The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

Performance before and after feature engineering was also compared.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Repository Structure

```
CreditWise/
│
├── loan_approval_predicter.ipynb
├── loan_approval_data.csv
└── README.md
```

---

## Getting Started

1. Clone this repository.

```bash
git clone https://github.com/your-username/CreditWise.git
```

2. Open the project in Jupyter Notebook or Google Colab.

3. Install any required Python libraries if they are not already installed.

4. Run the notebook cells sequentially to reproduce the results.

---

## Key Learning Outcomes

This project demonstrates practical experience with:

- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling
- Machine Learning Classification
- Model Evaluation
- End-to-End Machine Learning Workflow

---

## Future Improvements

Potential enhancements include:

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- ROC Curve and AUC analysis
- Random Forest Classifier
- XGBoost Classifier
- Model deployment using Streamlit or Flask
- Interactive prediction interface

---

## Results

- Successfully built and evaluated multiple classification models.
- Compared model performance using standard evaluation metrics.
- Improved model performance through feature engineering.
- Demonstrated a complete machine learning pipeline suitable for binary classification problems.

---

## Author

**Kunal Mohnani**

B.Tech Computer Science Engineering

Aspiring Data Scientist | Machine Learning Enthusiast

---

## License

This project is licensed under the MIT License.
