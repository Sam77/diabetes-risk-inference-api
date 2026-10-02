"""Train a diabetes risk classifier and save it for the inference API.

Approach mirrors my Master of Data Science capstone: missing-value handling,
min-max scaling, L1-regularised logistic regression, a 60/20/20
train/validation/test split, regularisation strength (C) tuned on the
validation set, and final evaluation on a held-out test set.
"""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

DATA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
            "master/pima-indians-diabetes.data.csv")
FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age"]
# In this dataset a value of 0 in these columns means "not measured"
ZERO_MEANS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
SEED = 42


def build_pipeline(c):
    impute = ColumnTransformer(
        [("impute", SimpleImputer(missing_values=0, strategy="median"), ZERO_MEANS_MISSING)],
        remainder="passthrough")
    return Pipeline([
        ("impute", impute),
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(penalty="l1", C=c, solver="liblinear",
                                     class_weight="balanced")),
    ])


def main():
    df = pd.read_csv(DATA_URL, header=None, names=FEATURES + ["outcome"])
    X, y = df[FEATURES], df["outcome"]

    # 60% train, 20% validation, 20% test, stratified to keep class balance
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    # Tune C on the validation set. Select on ROC AUC, and report recall too,
    # because missing a true high-risk case matters most in this domain.
    best_c, best_auc = None, -1.0
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
        candidate = build_pipeline(c).fit(X_train, y_train)
        auc = roc_auc_score(y_val, candidate.predict_proba(X_val)[:, 1])
        recall = recall_score(y_val, candidate.predict(X_val))
        print(f"C={c:<5} validation AUC={auc:.3f} recall={recall:.3f}")
        if auc > best_auc:
            best_c, best_auc = c, auc

    pipeline = build_pipeline(best_c).fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    metrics = {
        "test_auc": round(float(roc_auc_score(y_test, probs)), 3),
        "test_recall": round(float(recall_score(y_test, preds)), 3),
        "test_mcc": round(float(matthews_corrcoef(y_test, preds)), 3),
    }
    print(f"Selected C={best_c}; test metrics: {metrics}")

    joblib.dump(pipeline, "model/model.joblib")
    with open("model/metadata.json", "w") as f:
        json.dump({"algorithm": "LogisticRegression (L1, liblinear)",
                   "C": best_c, "features": FEATURES, **metrics}, f, indent=2)


if __name__ == "__main__":
    main()
