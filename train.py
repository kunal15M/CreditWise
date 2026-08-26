import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("loan_approval_data.csv")
df.dropna(subset=["Loan_Approved"], inplace=True)
X = df.drop(columns=["Loan_Approved", "Applicant_ID"])
Y = df["Loan_Approved"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

num_features = [
    'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
    'Collateral_Value', 'Loan_Amount', 'Loan_Term'
]

cat_features = [
    'Employment_Status', 'Marital_Status', 'Loan_Purpose',
    'Property_Area', 'Education_Level', 'Gender', 'Employer_Category'
]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

best_pipeline.fit(X_train, Y_train)

joblib.dump(best_pipeline, 'creditwise_pipeline.joblib')
print("Model pipeline exported successfully.")
