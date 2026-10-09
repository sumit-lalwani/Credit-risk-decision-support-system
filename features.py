"""Shared feature engineering and preprocessing for training and Flask inference.

The feature logic in this module is used by BOTH train_model.py and app.py so
that training data and website inputs are prepared in the same way.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

NUM_ATTRIBUTES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "monthly_income",
    "monthly_interest_burden",
    "credit_history_to_age",
    "monthly_interest_income_ratio",
    "income_loan_gap",
]

CAT_ATTRIBUTES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
]

BINARY_ATTRIBUTES = ["cb_person_default_on_file"]

RAW_INPUT_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def build_pipeline():
    """Build the same column-transforming structure used in the user's script."""
    num_pipeline = "passthrough"

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    default_file_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=[["N", "Y"]]))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, NUM_ATTRIBUTES),
        ("cat", cat_pipeline, CAT_ATTRIBUTES),
        ("default_file", default_file_pipeline, BINARY_ATTRIBUTES),
    ])


def learn_imputation_values(training_data: pd.DataFrame) -> dict:
    """Calculate fill values from training data only."""
    emp_median = float(training_data["person_emp_length"].median())
    overall_interest_median = float(training_data["loan_int_rate"].median())
    grade_interest_medians = (
        training_data.groupby("loan_grade")["loan_int_rate"].median().dropna().to_dict()
    )
    grade_interest_medians = {str(k): float(v) for k, v in grade_interest_medians.items()}

    return {
        "person_emp_length_median": emp_median,
        "loan_int_rate_median": overall_interest_median,
        "loan_int_rate_median_by_grade": grade_interest_medians,
    }


def apply_imputation(data: pd.DataFrame, imputation_values: dict) -> pd.DataFrame:
    """Fill the same two columns handled by the original training script."""
    data = data.copy()
    data["person_emp_length"] = data["person_emp_length"].fillna(
        imputation_values["person_emp_length_median"]
    )

    by_grade = data["loan_grade"].map(
        imputation_values["loan_int_rate_median_by_grade"]
    )
    data["loan_int_rate"] = data["loan_int_rate"].fillna(by_grade)
    data["loan_int_rate"] = data["loan_int_rate"].fillna(
        imputation_values["loan_int_rate_median"]
    )
    return data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the engineered columns used by this project.

    loan_percent_income is calculated here (loan amount / annual income), rather
    than asking website users to enter a redundant ratio manually.
    """
    data = data.copy()
    data["loan_percent_income"] = data["loan_amnt"] / data["person_income"]
    data["monthly_income"] = data["person_income"] / 12
    data["monthly_interest_burden"] = (
        data["loan_amnt"] * (data["loan_int_rate"] / 100) / 12
    )
    data["credit_history_to_age"] = (
        data["cb_person_cred_hist_length"] / data["person_age"]
    )
    data["monthly_interest_income_ratio"] = (
        data["monthly_interest_burden"] / data["monthly_income"]
    )
    data["income_loan_gap"] = data["person_income"] - data["loan_amnt"]
    return data
