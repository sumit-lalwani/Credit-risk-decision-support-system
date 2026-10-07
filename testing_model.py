import pandas as pd
import numpy as np
import joblib
import os 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE= "model.pkl"
PIPELINE_FILE= "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes, binary_attributes):

    # for numerical columns
    num_pipeline="passthrough"

    # For categoriacl columns
    cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])

    # For binary column
    default_file_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            categories=[["N", "Y"]]
        ))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes),
        ("default_file", default_file_pipeline, binary_attributes)
    ])

    return full_pipeline

def create_features(data):
    data["monthly_income"] = data["person_income"] / 12
    data["monthly_interest_burden"] = (data["loan_amnt"] * (data["loan_int_rate"] / 100) / 12)
    data["credit_history_to_age"] = (data["cb_person_cred_hist_length"] / data["person_age"])
    data["monthly_interest_income_ratio"] = (data["monthly_interest_burden"] /data["monthly_income"])
    data["income_loan_gap"] = data["person_income"] - data["loan_amnt"]

    return data

if not os.path.exists(MODEL_FILE):
    # 1. Load the data
    df=pd.read_excel("Credit risk dataset.xlsx")

    # 2. Drop the duplicates
    df=df.drop_duplicates().reset_index(drop=True)

    # 3. Split train and test data
    split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['loan_status']):
        strat_train_set= df.iloc[train_index]
        strat_test_set= df.iloc[test_index]

    # Making copy of data to protect original data
    risk= strat_train_set.copy()
    risk_labels= risk['loan_status'].copy()
    risk= risk.drop('loan_status', axis=1)

    test_risk= strat_test_set.copy()

    # 4.Filling missing values
    emp_median= risk['person_emp_length'].median()
    grade_median= risk.groupby("loan_grade")["loan_int_rate"].median()

    risk["person_emp_length"] = risk["person_emp_length"].fillna(emp_median)
    risk["loan_int_rate"] = risk["loan_int_rate"].fillna(risk["loan_grade"].map(grade_median))

    test_risk["person_emp_length"] = test_risk["person_emp_length"].fillna(emp_median)
    test_risk["loan_int_rate"] = test_risk["loan_int_rate"].fillna(test_risk["loan_grade"].map(grade_median))

    test_risk.to_csv("risk test data.csv" , index=False)

    # Feature engineering
    risk= create_features(risk)
    test_risk= create_features(test_risk)
    test_risk.to_csv("input test data.csv" , index=False)


    # List numerical columns, categorical columns and binary columns
    num_attributes= risk.drop(columns=['person_home_ownership','loan_intent', 'loan_grade', 'cb_person_default_on_file' ], axis=1).columns.tolist()
    cat_attributes= ['person_home_ownership', 'loan_intent', "loan_grade"]
    binary_attributes = ['cb_person_default_on_file']

    pipeline= build_pipeline(num_attributes, cat_attributes, binary_attributes)
    risk_prepared= pipeline.fit_transform(risk)
    

    model= RandomForestClassifier(random_state=42)
    model.fit(risk_prepared, risk_labels)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Congratulations, Your model is trained.")
else:
    model= joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE_FILE)
    test_risk=pd.read_csv("input test data.csv")
    original_columns= pd.read_csv("risk test data.csv").columns.to_list()
    test_risk_labels= test_risk['loan_status'].copy()
    test_risk= test_risk.drop('loan_status', axis=1)
    test_set_prepared= pipeline.transform(test_risk)
    predictions= model.predict(test_set_prepared)
    test_risk["loan_status"]=predictions
    default_probability= model.predict_proba(test_set_prepared)[:,1]
    default_probability= default_probability * 100
    output_file= test_risk[original_columns].copy()
    output_file["actual_loan_status"]= test_risk_labels
    output_file["predicted_loan_status"]= predictions
    output_file["predicted_default_probability"]= default_probability
    output_file.to_csv("output.csv" , index=False)
    print("Inference is complete, results saved to \"output.csv\". Enjoy!")