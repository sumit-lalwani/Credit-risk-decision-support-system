import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report )

# 1. Load the data
df= pd.read_excel("Credit risk dataset.xlsx")

# 2. Drop duplicates
df= df.drop_duplicates().reset_index(drop=True)
print(f"Shape of data after dropping duplicates: {df.shape}")

# 3. Create a stratified train and test set
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['loan_status']):
    strat_train_set= df.iloc[train_index]
    strat_test_set= df.iloc[test_index]

# We will work on copy of this dataset

risk= strat_train_set.copy()

# 4. Sepearte features and labels of both train and test set
risk_labels= risk['loan_status'].copy()
risk= risk.drop('loan_status', axis=1)

test_risk= strat_test_set.copy()
test_labels= test_risk['loan_status'].copy()
test_risk= test_risk.drop('loan_status', axis=1)

# 5.Filling missing values

emp_median= risk['person_emp_length'].median()
grade_median= risk.groupby("loan_grade")["loan_int_rate"].median()

risk["person_emp_length"] = risk["person_emp_length"].fillna(emp_median)
risk["loan_int_rate"] = risk["loan_int_rate"].fillna(risk["loan_grade"].map(grade_median))

test_risk["person_emp_length"] = test_risk["person_emp_length"].fillna(emp_median)
test_risk["loan_int_rate"] = test_risk["loan_int_rate"].fillna(test_risk["loan_grade"].map(grade_median))



# 6.Feature engineering
risk["monthly_income"] = risk["person_income"] / 12
risk["monthly_interest_burden"] = (risk["loan_amnt"] * (risk["loan_int_rate"] / 100) / 12)
risk["credit_history_to_age"] = (risk["cb_person_cred_hist_length"] / risk["person_age"])
risk["monthly_interest_income_ratio"] = (risk["monthly_interest_burden"] /risk["monthly_income"])
risk["income_loan_gap"] = risk["person_income"] - risk["loan_amnt"]

test_risk["monthly_income"] = test_risk["person_income"] / 12
test_risk["monthly_interest_burden"] = (test_risk["loan_amnt"] * (test_risk["loan_int_rate"] / 100) / 12)
test_risk["credit_history_to_age"] = (test_risk["cb_person_cred_hist_length"] / test_risk["person_age"])
test_risk["monthly_interest_income_ratio"] = (test_risk["monthly_interest_burden"] /test_risk["monthly_income"])
test_risk["income_loan_gap"] = test_risk["person_income"] - test_risk["loan_amnt"]

# 7.List numerical columns, categorical columns and binary columns
num_attributes= risk.drop(columns=['person_home_ownership','loan_intent', 'loan_grade', 'cb_person_default_on_file' ], axis=1).columns.tolist()
cat_attributes= ['person_home_ownership', 'loan_intent', "loan_grade"]
binary_attributes = ['cb_person_default_on_file']

# 8.Let's make the pipeline
# For numerical columns
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

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

# 9.Transform the data
risk_prepared= full_pipeline.fit_transform(risk)

risk_prepared = pd.DataFrame(
    risk_prepared,
    columns=full_pipeline.get_feature_names_out(),
    index=risk.index
)

risk_prepared.columns = (
    risk_prepared.columns
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
    .str.replace("default_file__", "", regex=False)
)

# 10. Preparing test data to evaluate model
test_risk_prepared= full_pipeline.transform(test_risk)
test_risk_prepared = pd.DataFrame(
    test_risk_prepared,
    columns=full_pipeline.get_feature_names_out(),
    index=test_risk.index
)

test_risk_prepared.columns = (
    test_risk_prepared.columns
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
    .str.replace("default_file__", "", regex=False)
)
# 10. Train the models and evaluate them by different metrices to find which one is better

# Logistic regression
print("Logistic regression.....")
log_reg= LogisticRegression()
log_reg.fit(risk_prepared, risk_labels)
log_pred= log_reg.predict(test_risk_prepared)
accuracy= accuracy_score(test_labels, log_pred)
print(f"Accuracy: {accuracy}")
precison= precision_score(test_labels, log_pred)
print(f"Precison score is : {precison}")
recall= recall_score(test_labels,log_pred)
print(f"Recall score is : {recall}")
f1= f1_score(test_labels,log_pred)
print(f"F1 score is : {f1}")
# roc= roc_auc_score(test_labels,log_pred)
# print(f"Roc Auc score is : {roc}")
cm= confusion_matrix(test_labels,log_pred)
print(f"Confusion matrix : {cm}")
print("\n")

print("Decision tree.........")
dec_tree= DecisionTreeClassifier()
dec_tree.fit(risk_prepared, risk_labels)
dec_pred= dec_tree.predict(test_risk_prepared)
accuracy= accuracy_score(test_labels, dec_pred)
print(f"Accuracy: {accuracy}")
precison= precision_score(test_labels, dec_pred)
print(f"Precison score is : {precison}")
recall= recall_score(test_labels,dec_pred)
print(f"Recall score is : {recall}")
f1= f1_score(test_labels,dec_pred)
print(f"F1 score is : {f1}")
# roc= roc_auc_score(test_labels,dec_pred)
# print(f"Roc Auc score is : {roc}")
cm= confusion_matrix(test_labels,dec_pred)
print(f"Confusion matrix : {cm}")
print("\n")

print("Random forest.....")
rand_forest= RandomForestClassifier()
rand_forest.fit(risk_prepared, risk_labels)
rand_pred= rand_forest.predict(test_risk_prepared)
accuracy= accuracy_score(test_labels, rand_pred)
print(f"Accuracy: {accuracy}")
precison= precision_score(test_labels, rand_pred)
print(f"Precison score is : {precison}")
recall= recall_score(test_labels,rand_pred)
print(f"Recall score is : {recall}")
f1= f1_score(test_labels,rand_pred)
print(f"F1 score is : {f1}")
# roc= roc_auc_score(test_labels,rand_pred)
# print(f"Roc Auc score is : {roc}")
cm= confusion_matrix(test_labels,rand_pred)
print(f"Confusion matrix : {cm}")
print("\n")

print("Gradient boosting.....")
grad_bost= GradientBoostingClassifier()
grad_bost.fit(risk_prepared, risk_labels)
grad_bost_pred= grad_bost.predict(test_risk_prepared)
accuracy= accuracy_score(test_labels, grad_bost_pred)
print(f"Accuracy: {accuracy}")
precison= precision_score(test_labels, grad_bost_pred)
print(f"Precison score is : {precison}")
recall= recall_score(test_labels,grad_bost_pred)
print(f"Recall score is : {recall}")
f1= f1_score(test_labels,grad_bost_pred)
print(f"F1 score is : {f1}")
# roc= roc_auc_score(test_labels,grad_bost_pred)
# print(f"Roc Auc score is : {roc}")
cm= confusion_matrix(test_labels,grad_bost_pred)
print(f"Confusion matrix : {cm}")
print("\n")

print("Probability of default")
default_probability= rand_forest.predict_proba(test_risk_prepared)[:,1]
default_probability= default_probability * 100
print(default_probability)