# Credit Risk Prediction Project

A learning project that uses historical loan data to estimate loan-default risk. It includes the original Python modeling script, an updated end-to-end architecture for a Flask web app, and a short PDF summary of the model evaluation. **Power BI has been removed; the other project components remain in the planned architecture, including Excel logging.**

## Project workflow

1. Load the historical loan dataset from Excel.
2. Inspect missing values, class balance, distributions, relationships, and exact duplicate rows; remove full-row duplicates.
3. Engineer financial and credit-history features.
4. Split the dataset into training and test sets, and learn preprocessing values from the training set.
5. Encode categorical fields and train/evaluate the classification model.
6. Let a new applicant enter information through an HTML/CSS form.
7. Use Flask to validate the input, calculate the same engineered features, and pass the input through the saved preprocessing pipeline and model.
8. Show the predicted class and estimated probability of default.
9. Append the application and its prediction to an Excel log without overwriting previous rows.

See [`docs/architecture.md`](docs/architecture.md) for the architecture diagram and further detail.

## Files in this package

- `original_code/testing_model.py` — preserved copy of the original modeling script.
- `docs/architecture.md` — project architecture with Power BI removed.
- `credit_risk_results_summary.pdf` — brief results, confusion matrix, conclusion, and limitations.

**Important:** this package contains the original modeling script and project documentation, but not the complete Flask app source code. When assembling the final repository, add your actual Flask, HTML template, CSS, and dependency files, then update the run instructions below if their names or locations differ.

## Run the original Python model

Place `testing_model.py` and `Credit risk dataset.xlsx` in the same working folder. Install dependencies:

```bash
python -m pip install pandas numpy scikit-learn openpyxl joblib
```

Run the script:

```bash
python testing_model.py
```

The script loads the Excel dataset, removes exact duplicate rows, makes a stratified 80/20 train-test split, imputes missing values using training-set statistics, engineers features, encodes fields, and trains a `RandomForestClassifier`. It saves model/preprocessing artifacts. **In the current script flow, the first run trains and saves the model; a subsequent run enters the prediction/export branch and creates a CSV output.** Run it from the folder where the dataset path resolves.

### Features engineered in the original script

- `monthly_income`
- `monthly_interest_burden`
- `credit_history_to_age`
- `monthly_interest_income_ratio`
- `income_loan_gap`

The script uses one-hot encoding for categorical fields and ordinal encoding for `cb_person_default_on_file` (`N`/`Y`).

## Run the Flask web app

The web-app workflow should use the same engineered features and fitted preprocessing as the training code. Once the actual app files have been added to the repository:

1. Open a terminal in the folder containing the app's `requirements.txt`.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Train/save the model artifacts using the training command defined by your project (for example, `python train_model.py`), if required.
4. Start Flask using the app's entry-point command (for example, `python app.py`).
5. Open `http://127.0.0.1:5000` in your browser.

These filenames and commands are examples until they are matched to the actual Flask files. The live prediction path should validate the submitted data, apply the same feature engineering and saved preprocessing/model, display the prediction and estimated default probability, and append a new record to Excel while retaining prior rows. Do not claim Excel logging is operational until it has been implemented and tested in the app.

## Evaluation results

On the reproduced stratified 80/20 holdout split, the original Random Forest workflow produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 93.35% |
| Precision for default class (1) | 96.25% |
| Recall for default class (1) | 72.43% |
| F1-score for default class (1) | 82.66% |
| ROC-AUC | 93.31% |

Confusion matrix (actual rows, predicted columns):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 5,026 | 40 |
| Actual 1 | 391 | 1,027 |

The model identified 1,027 of 1,418 actual defaults and missed 391 in this test split. Accuracy alone is not enough for credit-risk evaluation: missed defaults (false negatives) matter, so default-class recall and the costs of different errors should be reviewed when selecting a decision threshold. These figures describe one holdout test and do not prove that the model is ready for real lending decisions. See `credit_risk_results_summary.pdf` for a recruiter-oriented summary.

## Limitations and responsible use

- The original script uses one Random Forest configuration rather than a systematic comparison of multiple algorithms.
- Evaluation uses a single stratified holdout split, not external or time-based validation.
- The probability-column selection should be made robust by checking `model.classes_` instead of assuming class `1` is always column index `1`.
- A probability estimate is not a guaranteed outcome or an automatic loan approval decision.
- This is an educational demonstration, not a validated production credit-decision system.
- Before publishing, verify that you are permitted to share the dataset. Do not commit real personal data, credentials, secrets, or confidential applicant information.
