# Credit Risk Project Architecture (Power BI removed)

## End-to-end flow

1. **Historical loan dataset** — Load the historical credit-risk Excel dataset.
2. **Data cleaning** — Inspect missing values and exact duplicate rows; remove full-row duplicates.
3. **Feature engineering** — Create monthly income, monthly interest burden, credit-history-to-age ratio, monthly-interest-to-income ratio, and income-loan gap.
4. **EDA and statistical analysis** — Examine class balance, missingness, distributions, relationships, and risk patterns before modeling.
5. **Preprocessing and model training** — Split data into training and test sets; learn imputation values on training data; encode categorical features; train and compare classification models.
6. **Model evaluation** — Report accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix. Pay particular attention to recall for defaults.
7. **New applicant website** — Applicant enters details in the HTML/CSS form.
8. **Flask backend** — Validate inputs, calculate the same engineered features, and apply the saved preprocessing pipeline and trained model.
9. **Prediction output** — Show the predicted class and estimated probability of default with an explanation that it is a model estimate, not an automatic lending decision.
10. **Excel application log** — Append the new application and prediction to an Excel sheet while preserving existing rows for later review.

**Removed:** Power BI/dashboard integration only. All other project components and the Excel logging step remain in the architecture.

## Simplified architecture diagram

```text
Historical Excel Dataset
          |
          v
Cleaning + EDA + Statistical Analysis
          |
          v
Feature Engineering + Preprocessing
          |
          v
Train / Compare / Evaluate ML Models
          |
          v
Saved Model + Saved Preprocessing
          |
          +------------------------------+
                                         |
New Applicant -> HTML/CSS Form -> Flask Backend
                                         |
                                         v
                              Feature Engineering
                                         |
                                         v
                              Saved Pipeline + Model
                                         |
                                         v
                          Default Class + Default Probability
                                         |
                                         v
                          Append Application to Excel Log
```

## Key implementation rule

Training and inference must use the same feature definitions, column order, encoders, and training-derived imputation values. Do not fit preprocessing again on a single new applicant. Excel logging should append rows rather than overwrite historical records.
