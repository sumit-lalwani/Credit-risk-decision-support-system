"""Train the Random Forest model and save artifacts for the Flask website."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from features import apply_imputation, build_pipeline, create_features, learn_imputation_values

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Credit risk dataset.xlsx"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Put 'Credit risk dataset.xlsx' in the data folder."
        )

    print("Loading dataset...")
    df = pd.read_excel(DATA_FILE)
    starting_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {starting_rows - len(df)} duplicate rows. Remaining rows: {len(df)}")

    X = df.drop(columns=["loan_status"]).copy()
    y = df["loan_status"].astype(int).copy()

    # Split before learning imputation values, so held-out data does not inform training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputation_values = learn_imputation_values(X_train)
    X_train = apply_imputation(X_train, imputation_values)
    X_test = apply_imputation(X_test, imputation_values)

    # This shared function is also called by app.py for new applicants.
    X_train = create_features(X_train)
    X_test = create_features(X_test)

    preprocessing = build_pipeline()
    X_train_prepared = preprocessing.fit_transform(X_train)
    X_test_prepared = preprocessing.transform(X_test)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    print("Training Random Forest...")
    model.fit(X_train_prepared, y_train)

    predictions = model.predict(X_test_prepared)
    default_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test_prepared)[:, default_class_index]

    print("\nHeld-out test results (not a lending-policy approval decision):")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(preprocessing, MODEL_DIR / "pipeline.pkl")
    joblib.dump(imputation_values, MODEL_DIR / "imputation.pkl")

    print(f"\nSaved model artifacts to: {MODEL_DIR}")
    print("Now run: python app.py")


if __name__ == "__main__":
    main()
