# Diabetes Risk Inference API

A small, production-style system that trains a machine learning model and serves it through a REST inference API. The model predicts the risk of diabetes from eight clinical measurements and returns a probability for each patient record.

The project shows the full path from a trained model to a running service: training and evaluation, saving the model as an artefact, loading it once at server startup, validating input, serving single and batch predictions, automated tests, a Docker image, and a CI pipeline.

The modelling approach follows my Master of Data Science (RMIT University) capstone. That project predicted future type 2 diabetes risk using the AusDiab dataset, and an L1-regularised logistic regression model performed best. The capstone data is not public, so this project applies the same approach to a public dataset.

## How it works

```
train.py ──► model/model.joblib + model/metadata.json ──► app.py (FastAPI) ──► JSON predictions
   │                                                          │
   └── download data, impute, scale, tune, evaluate           └── validate input, predict_proba
```

1. `train.py` downloads the dataset and builds a scikit-learn `Pipeline`:
   - **Missing values:** zeros in glucose, blood pressure, skin thickness, insulin and BMI mean "not measured", so they are replaced with the training median (`SimpleImputer`).
   - **Scaling:** `MinMaxScaler`.
   - **Model:** `LogisticRegression` with an L1 (Lasso) penalty, the `liblinear` solver, and balanced class weights.
2. The data is split 60/20/20 into training, validation and test sets, stratified by outcome.
3. The regularisation strength `C` is tuned on the validation set using ROC AUC. Recall is reported as well, because missing a true high-risk case matters most in this domain.
4. The final model is evaluated once on the held-out test set (ROC AUC, recall, Matthews correlation coefficient). The whole pipeline is saved, so the server applies exactly the same preprocessing as training.
5. `app.py` loads the pipeline once at startup and serves predictions.

## Dataset

[Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv): 768 records, 8 numeric features, binary outcome (diabetes onset within five years). Originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and whether the model is loaded |
| GET | `/model-info` | Algorithm, selected `C`, feature list and test metrics |
| POST | `/predict` | Risk prediction for one patient |
| POST | `/predict/batch` | Risk predictions for 1 to 1000 patients |

Interactive documentation is generated automatically at `http://localhost:8000/docs`.

Input is validated with Pydantic. Out-of-range or missing fields return HTTP 422 instead of reaching the model.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
       "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}'
```

### Example response

```json
{"risk_probability": 0.689, "high_risk": true}
```

## Running locally

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
python train.py                  # trains the model and writes model/
uvicorn app:app --reload         # serves the API on http://localhost:8000
pytest -v                        # runs the API tests
```

## Running with Docker

The image trains the model during the build, so the container starts with a ready model.

```bash
docker build -t diabetes-risk-api .
docker run -p 8000:8000 diabetes-risk-api
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. It installs dependencies, trains the model and runs the test suite.

## Results

On the held-out test set (154 records), the selected model (`C = 1.0`) achieved:

| Metric | Value |
|---|---|
| ROC AUC | 0.787 |
| Recall | 0.66 |
| Matthews correlation coefficient | 0.381 |

## Project structure

```
├── train.py              # data loading, pipeline, tuning, evaluation, model export
├── app.py                # FastAPI inference server
├── tests/test_app.py     # API tests (health, single, batch, invalid input)
├── model/                # trained model artefact and metadata (generated)
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Limitations and next steps

- This is a demonstration project, not a clinical tool, and must not be used for medical decisions.
- The 0.5 decision threshold is a default. In practice it would be set with domain experts to balance recall against false positives.
- Possible extensions: compare a random forest model, track experiments and model versions with MLflow, add request logging and monitoring, and deploy to Kubernetes.

## Author

Sam Rathnayake ([LinkedIn](https://www.linkedin.com/in/sam-rathnayake/))
