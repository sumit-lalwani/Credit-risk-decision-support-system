# Credit Risk Analysis & Loan Default Prediction

A machine learning project for analysing credit risk and predicting the probability of loan default using borrower and loan information.

The project currently focuses on **data preparation, feature engineering, machine learning model evaluation, Random Forest training, and default probability prediction**.

> **Current stage:** Machine Learning Model Development
> **Next stage:** Backend + New Applicant Prediction Website

---

## Project Objective

The main objective of this project is to build a credit-risk prediction system that can:

* Analyse borrower and loan characteristics
* Identify factors associated with loan default
* Compare multiple machine learning classification models
* Select a suitable model for default prediction
* Predict whether a borrower is likely to default
* Calculate the **probability of default**
* Save the trained model and preprocessing pipeline for future deployment

---

## Dataset

The project uses a credit-risk dataset containing borrower, loan, and credit-history information.

### Target Variable

`loan_status`

```text
0 → No Default
1 → Default
```

### Main Input Features

* `person_age`
* `person_income`
* `person_home_ownership`
* `person_emp_length`
* `loan_intent`
* `loan_grade`
* `loan_amnt`
* `loan_int_rate`
* `loan_percent_income`
* `cb_person_default_on_file`
* `cb_person_cred_hist_length`

---

## Project Workflow

```text
Raw Dataset
     ↓
Duplicate Removal
     ↓
Train / Test Split
     ↓
Missing Value Handling
     ↓
Feature Engineering
     ↓
Data Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Random Forest Selection
     ↓
Default Probability Prediction
     ↓
Save Model + Pipeline
     ↓
Backend Development
```

---

# Files

## `model_evaluation.py`

This script is responsible for evaluating different machine learning models.

### Main operations

* Load the dataset
* Remove duplicate rows
* Create a stratified train/test split
* Separate features and target
* Handle missing values
* Perform feature engineering
* Encode categorical variables
* Prepare training and testing data
* Train multiple classification models
* Evaluate model performance
* Generate confusion matrices
* Calculate probability of default for the selected Random Forest model

### Models evaluated

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### Evaluation Metrics

The models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Probability-based metrics can also be added as the model evaluation is further refined.

---

## `testing_model.py`

This script is used to train and save the Random Forest model and perform inference using the saved model.

### Main operations

* Load and clean the dataset
* Remove duplicate rows
* Create a stratified train/test split
* Apply missing-value rules
* Create engineered features
* Build the preprocessing pipeline
* Train the Random Forest classifier
* Save the trained model
* Save the preprocessing pipeline
* Load the saved model for inference
* Predict loan status
* Calculate probability of default
* Save prediction results

---

# Feature Engineering

Additional features are created from the existing variables to provide the model with more useful financial relationships.

### Monthly Income

```text
monthly_income = person_income / 12
```

### Monthly Interest Burden

```text
monthly_interest_burden =
loan_amnt × interest_rate / 100 / 12
```

### Credit History to Age

```text
credit_history_to_age =
credit_history_length / person_age
```

### Monthly Interest Income Ratio

```text
monthly_interest_income_ratio =
monthly_interest_burden / monthly_income
```

### Income Loan Gap

```text
income_loan_gap =
person_income - loan_amnt
```

---

# Data Preprocessing

### Duplicate Handling

Exact duplicate rows are removed before modelling.

### Missing Values

Current rules include:

* `person_emp_length` → median employment length calculated from the training data
* `loan_int_rate` → median interest rate based on `loan_grade`

These rules are applied consistently to the test data.

### Categorical Encoding

Categorical variables are converted into numerical values using `OneHotEncoder`.

### Binary Encoding

`cb_person_default_on_file` is converted using:

```text
N → 0
Y → 1
```

### Numerical Features

Since the selected final model is Random Forest, numerical features are passed through without standardisation because tree-based models do not require feature scaling.

---

# Model Selection

The project evaluates multiple models before selecting the final model.

The current Random Forest model showed the strongest overall performance among the evaluated models based on the current test results.

### Current test results

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     86.89% |     76.54% |     57.76% |     65.84% |
| Decision Tree       |     88.85% |     73.40% | **76.87%** |     75.09% |
| **Random Forest**   | **93.55%** | **96.90%** |     72.85% | **83.17%** |
| Gradient Boosting   |     92.80% |     94.40% |     71.30% |     81.24% |

Random Forest is currently being used as the selected model for the next stage.

---

# Default Probability

The Random Forest model provides a probability for each class.

```python
model.predict_proba(X)[:, 1]
```

Because:

```text
0 = No Default
1 = Default
```

the probability of class `1` is treated as the **probability of default**.

Example:

```text
Predicted Default Probability = 72.35%
```

This gives a more informative output than only predicting `0` or `1`.

---

# Saved Model Files

The current training workflow saves:

```text
model.pkl
pipeline.pkl
```

### `model.pkl`

Contains the trained Random Forest classifier.

### `pipeline.pkl`

Contains the preprocessing configuration required to transform data into the format expected by the model.

These files will be reused when building the backend.

---

# Current Output

The inference stage produces prediction data containing:

* Original applicant information
* Actual loan status
* Predicted loan status
* Predicted default probability

Example:

| loan_status | predicted_loan_status | predicted_default_probability |
| ----------: | --------------------: | ----------------------------: |
|           1 |                     1 |                        82.35% |
|           0 |                     0 |                        12.47% |
|           1 |                     1 |                        68.92% |

---

# Project Structure

```text
Credit-Risk-Analysis/
│
├── Credit risk dataset.xlsx
│
├── model_evaluation.py
├── testing_model.py
│
├── model.pkl
├── pipeline.pkl
│
├── risk test data.csv
├── input test data.csv
├── output.csv
│
└── README.md
```

---

# How to Run

## 1. Install dependencies

```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

## 2. Run model evaluation

```bash
python model_evaluation.py
```

This evaluates the different machine learning models.

## 3. Train / load the Random Forest model

```bash
python testing_model.py
```

This trains the model if the saved model files do not exist. Otherwise, it loads the saved model and performs inference.

---

# Current Status

### Completed

* Data cleaning
* Duplicate removal
* Missing value handling
* Feature engineering
* Stratified train/test splitting
* Data preprocessing
* Logistic Regression evaluation
* Decision Tree evaluation
* Random Forest evaluation
* Gradient Boosting evaluation
* Confusion matrix analysis
* Random Forest selection
* Default probability calculation
* Model saving
* Preprocessing pipeline saving

### Next Stage

The next stage is to build a **backend application** that accepts a new applicant's information instead of using a predefined test CSV.

Planned flow:

```text
New Applicant
     ↓
Backend Input
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Saved Preprocessing Pipeline
     ↓
Saved Random Forest Model
     ↓
Default Probability
     ↓
Prediction / Risk Output
```

The README will be expanded after the backend and website are completed.
