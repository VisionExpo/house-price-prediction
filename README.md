# Ames House Price Prediction Demo

A local, recruiter-facing web demonstration of the analysis in [`notebooks/PRCP-1020_House_Price_Prediction.ipynb`](notebooks/PRCP-1020_House_Price_Prediction.ipynb). The app uses FastAPI with server-rendered HTML, CSS, and vanilla JavaScript. It recreates the notebook's preprocessing steps and Gradient Boosting configuration; it does not alter the notebook or its saved results.

This is an educational portfolio demo, not a live-market valuation service.

## What the demo shows

- A house feature form with an interactive prediction from the fitted Gradient Boosting pipeline.
- The notebook's validation metrics for Linear Regression, Random Forest, and Gradient Boosting.
- Data and model notes, top reported feature importances, and limitations.
- An explicit note that the form accepts 8 of 79 predictors. The other 71 are populated from the training split's numeric medians and categorical most-frequent values. Each prediction is therefore illustrative and depends on those defaults.

## Data setup

The Ames CSV is not checked into Git because `data/raw/` is excluded by `.gitignore`. Place the assignment dataset at:

```text
data/raw/extracted/Data/data.csv
```

The current local workspace already has the CSV at that path. If you are setting up a fresh checkout and have the supplied zip archive, extract it into `data/raw/extracted/` so the CSV appears at the path above. Keep the assignment archive's `Data/data.csv` file name and directory casing.

## Install and run

Python 3.12 and `uv` are used by this repository.

```powershell
uv sync --dev
uv run uvicorn house_price.main:app --reload
```

Open `http://127.0.0.1:8000`. The first prediction fits the model in memory using the notebook's 80/20 split and `random_state=42`; later predictions reuse the fitted pipeline while the app process is running. The app does not save a model artifact.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`, and a basic status check is available at `/api/health`.

## Run the tests

```powershell
uv run pytest
```

The API integration test fits the real notebook-aligned pipeline using the local Ames CSV.

## Model and evaluation

The notebook uses median imputation for numerical columns and most-frequent imputation followed by one-hot encoding for categorical columns. The displayed Gradient Boosting configuration has 200 estimators, learning rate 0.05, max depth 3, squared-error loss, and random state 42.

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Linear Regression | $20,466.13 | $31,294.99 | 0.8723 |
| Random Forest | $17,350.76 | $28,376.65 | 0.8950 |
| Gradient Boosting | $16,946.24 | $27,667.91 | 0.9002 |

These figures are copied from the completed notebook's saved validation results. MAE and RMSE are in dollars. They describe one held-out split of 292 Ames homes and should not be read as guarantees for future sales. The app recreates the training pipeline for interactive estimates; it does not claim the live demo's output is independently validated against a market feed.

The notebook reports rounded feature importances of about 51% for `OverallQual` and 15% for `GrLivArea`. Feature importance describes model behavior, not causation.

## Recruiter demo flow

1. Start the app and open the estimate form.
2. Submit the prefilled example once to show a prediction coming from the real pipeline.
3. Change living area, overall quality, or neighborhood and submit again to demonstrate the interaction.
4. Point out the disclosure: only 8 of 79 model features are entered, and the remaining features use training-split defaults.
5. Compare the three validation rows and explain why the notebook selected Gradient Boosting for the interactive demonstration.
6. Walk through the preparation steps, 80/20 split, dataset summary, feature-importance caveat, and the model's limitations.

## Project structure

```text
house_price/       FastAPI routes, validation schemas, and model pipeline
templates/         Server-rendered HTML page
static/            CSS, browser interaction, and generated house illustration
tests/             API smoke and integration tests
notebooks/         Original analysis notebook and saved results
data/raw/          Local assignment data; ignored by Git
```
