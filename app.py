"""Flask web app for the Credit Risk / Default Probability demo."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from features import apply_imputation, create_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "model.pkl"
PIPELINE_FILE = MODEL_DIR / "pipeline.pkl"
IMPUTATION_FILE = MODEL_DIR / "imputation.pkl"

app = Flask(__name__)


def load_artifacts():
    missing = [str(p.name) for p in [MODEL_FILE, PIPELINE_FILE, IMPUTATION_FILE] if not p.exists()]
    if missing:
        raise RuntimeError(
            "Model files are missing: " + ", ".join(missing) +
            ". Run 'python train_model.py' first."
        )
    return (
        joblib.load(MODEL_FILE),
        joblib.load(PIPELINE_FILE),
        joblib.load(IMPUTATION_FILE),
    )


try:
    model, preprocessing, imputation_values = load_artifacts()
    ARTIFACTS_ERROR = None
except Exception as exc:
    model = preprocessing = imputation_values = None
    ARTIFACTS_ERROR = str(exc)


OWNERSHIP_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT_OPTIONS = [
    "EDUCATION", "MEDICAL", "PERSONAL", "VENTURE",
    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
]
GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_OPTIONS = ["N", "Y"]


def form_values():
    return {key: request.form.get(key, "") for key in [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
        "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]}


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        error=ARTIFACTS_ERROR,
        values={},
        ownership_options=OWNERSHIP_OPTIONS,
        intent_options=INTENT_OPTIONS,
        grade_options=GRADE_OPTIONS,
        default_options=DEFAULT_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = form_values()
    if ARTIFACTS_ERROR:
        return render_template(
            "index.html", error=ARTIFACTS_ERROR, values=values,
            ownership_options=OWNERSHIP_OPTIONS, intent_options=INTENT_OPTIONS,
            grade_options=GRADE_OPTIONS, default_options=DEFAULT_OPTIONS,
        ), 503

    try:
        age = int(values["person_age"])
        annual_income = float(values["person_income"])
        ownership = values["person_home_ownership"]
        employment_text = values["person_emp_length"].strip()
        employment_length = float(employment_text) if employment_text else None
        intent = values["loan_intent"]
        grade = values["loan_grade"]
        loan_amount = float(values["loan_amnt"])
        interest_text = values["loan_int_rate"].strip()
        interest_rate = float(interest_text) if interest_text else None
        previous_default = values["cb_person_default_on_file"]
        credit_history = int(values["cb_person_cred_hist_length"])

        if not 18 <= age <= 100:
            raise ValueError("Enter an age between 18 and 100.")
        if annual_income <= 0:
            raise ValueError("Annual income must be greater than zero.")
        if loan_amount <= 0:
            raise ValueError("Loan amount must be greater than zero.")
        if employment_length is not None and not 0 <= employment_length <= age:
            raise ValueError("Employment length must be between 0 and the applicant's age.")
        if interest_rate is not None and not 0 < interest_rate <= 100:
            raise ValueError("Interest rate must be greater than 0 and no more than 100%.")
        if not 0 <= credit_history <= age:
            raise ValueError("Credit history length must be between 0 and the applicant's age.")
        if ownership not in OWNERSHIP_OPTIONS:
            raise ValueError("Select a valid home ownership category.")
        if intent not in INTENT_OPTIONS:
            raise ValueError("Select a valid loan purpose.")
        if grade not in GRADE_OPTIONS:
            raise ValueError("Select a valid loan grade.")
        if previous_default not in DEFAULT_OPTIONS:
            raise ValueError("Select N or Y for previous default on file.")

        # Assemble raw columns first. Optional fields use the same training-derived
        # imputation values as the training script. Engineered features come next.
        applicant = pd.DataFrame([{
            "person_age": age,
            "person_income": annual_income,
            "person_home_ownership": ownership,
            "person_emp_length": employment_length,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_percent_income": None,  # calculated consistently by create_features()
            "cb_person_default_on_file": previous_default,
            "cb_person_cred_hist_length": credit_history,
        }])

        applicant = apply_imputation(applicant, imputation_values)
        applicant = create_features(applicant)
        applicant_prepared = preprocessing.transform(applicant)

        classes = list(model.classes_)
        default_index = classes.index(1)
        default_probability = float(model.predict_proba(applicant_prepared)[0, default_index])
        threshold = 0.50  # Demonstration threshold only; not a lender approval policy.
        predicted_status = int(default_probability >= threshold)

        return render_template(
            "result.html",
            probability=round(default_probability * 100, 2),
            predicted_status=predicted_status,
            status_label="Predicted default" if predicted_status == 1 else "Predicted non-default",
            threshold=int(threshold * 100),
            applicant={
                "age": age,
                "annual_income": f"{annual_income:,.2f}",
                "home_ownership": ownership.replace("_", " ").title(),
                "employment_length": (
                    f"{employment_length:g} years" if employment_length is not None
                    else f"Imputed using training median ({imputation_values['person_emp_length_median']:.1f} years)"
                ),
                "loan_intent": intent.replace("_", " ").title(),
                "loan_grade": grade,
                "loan_amount": f"{loan_amount:,.2f}",
                "interest_rate": (
                    f"{interest_rate:.2f}%" if interest_rate is not None
                    else f"Imputed from Grade {grade} history ({applicant['loan_int_rate'].iloc[0]:.2f}%)"
                ),
                "loan_to_income": f"{applicant['loan_percent_income'].iloc[0] * 100:.2f}%",
                "previous_default": "Yes" if previous_default == "Y" else "No",
                "credit_history": f"{credit_history} years",
            },
        )

    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html", error=str(exc), values=values,
            ownership_options=OWNERSHIP_OPTIONS, intent_options=INTENT_OPTIONS,
            grade_options=GRADE_OPTIONS, default_options=DEFAULT_OPTIONS,
        ), 400
    except Exception as exc:
        # Show a safe, actionable message rather than a stack trace in the page.
        return render_template(
            "index.html",
            error=("Prediction could not be completed. Check that the model was trained "
                   "with the files in this project, then run train_model.py again."),
            values=values,
            ownership_options=OWNERSHIP_OPTIONS, intent_options=INTENT_OPTIONS,
            grade_options=GRADE_OPTIONS, default_options=DEFAULT_OPTIONS,
        ), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
