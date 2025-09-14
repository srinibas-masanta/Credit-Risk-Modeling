# Credit Risk Modeling

## Project Overview

This repository contains an end-to-end credit risk modeling project using the **German Credit** dataset. The goal is to build, evaluate, and deploy a classifier that predicts whether a loan applicant is a **good** or **bad** credit risk. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model training and tuning (Decision Tree, Random Forest, Extra Trees, XGBoost), model selection, and a Streamlit web app for instant prediction.

Key takeaways:

* Dataset: **1,000** records with 11 columns
* Target distribution: **700 good / 300 bad** (70% / 30%)
* Final selected model: **Random Forest** (saved as `random_forest_credit_model.pkl`)
* Deployable Streamlit app: `app.py`

---


## Dataset

The dataset file used in this project is `german_credit_data.csv`. Important notes about the data:

* **Total rows:** 1000
* **Columns of interest:** `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`, `Risk`.
* **Missing values:** `Saving accounts` (183 missing, \~18.3%), `Checking account` (394 missing, \~39.4%).

Handling of missing values:

* For `Saving accounts` and `Checking account` the missing entries were imputed with the string **`"unknown"`**. For financial account features, missingness often carries information (e.g., a missing checking account may mean the applicant does not have one).
* `Unnamed: 0` (index column) was dropped.

---

## Exploratory Data Analysis (EDA) & Insights

A thorough EDA was performed using histograms, boxplots, countplots, scatterplots, violin plots, and a correlation heatmap. Key insights:

* **Age:** Average applicant age \~35.5 (min 19, max 75). Bad applicants slightly younger on average (33.96) than good applicants (36.22).
* **Credit amount:** Strong right skew — a small number of applicants request much larger loans. Bad applicants borrow more on average (\~3938) vs good (\~2985).
* **Duration:** Bad applicants have longer average loan durations (\~25 months) vs good (\~19 months).
* **Job:** Job category `3` requests substantially higher average loans (\~5435) vs others.
* **Sex:** Males request higher average credit amounts (\~3448) than females (\~2878).
* **Feature relationships:** `Credit amount` and `Duration` are strongly correlated (corr ≈ 0.62).

These EDA results guided feature selection and transformation decisions.

---

## Preprocessing & Feature Engineering

Main preprocessing steps:

1. **Dropped** `Unnamed: 0` (index).
2. **Imputed** `Saving accounts` and `Checking account` with the category `"unknown"`.
3. **Selected features** used for modeling:

   ```text
   ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']
   ```
4. **Target:** `Risk` (label encoded). Mapping used in training: `good -> 1`, `bad -> 0`.
5. **Encoding:** LabelEncoder was used for categorical variables (`Sex`, `Housing`, `Saving accounts`, `Checking account`) and the target. All encoders were saved with `joblib` (e.g. `Sex_encoder.pkl`).
6. **Train/Test split:** 80/20 split with `stratify=y`, `random_state=42`.
7. **Imbalance handling:** SMOTE was applied on the training set to balance classes before hyperparameter tuning.

Notes:

* LabelEncoding was chosen here for its simplicity given the small number of categories; for future work, `OneHotEncoder` or `OrdinalEncoder` inside a `ColumnTransformer` + `Pipeline` is recommended for production robustness.

---

## Modeling & Hyperparameter Tuning

Models evaluated:

* `DecisionTreeClassifier` (class\_weight='balanced')
* `RandomForestClassifier` (class\_weight='balanced')
* `ExtraTreesClassifier` (class\_weight='balanced')
* `XGBClassifier`

Hyperparameter search:

* Used `GridSearchCV` (5-fold CV, scoring=`accuracy`) to find the best params for each model.
* `SMOTE` was applied to training folds (training pipeline order: split → SMOTE on train → GridSearchCV on resampled train).

**Best parameters found (summary):**

* Decision Tree: `{'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}`
* Random Forest: `{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2}`
* Extra Trees: `{'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}`
* XGBoost: `{'colsample_bytree': 0.7, 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.7}`

Evaluation metrics during selection: accuracy, precision, recall, F1-score, ROC-AUC. The experiments were reproducible with `random_state=42` where applicable.

---

## Evaluation & Model Selection

All four models were trained and evaluated on the held-out test set (20% of data). Below are the most important evaluation results:

* **Decision Tree** — Accuracy: **0.665**
* **Random Forest** — Accuracy: **0.735**, ROC-AUC: **0.7714**
* **Extra Trees** — Accuracy: **0.715**
* **XGBoost** — Accuracy: **0.735**, ROC-AUC: **0.7525**

**Confusion matrices (test set):**

* Random Forest:

```
[[ 43  17]
 [ 36 104]]
```

(Row 0 = true `bad`, Row 1 = true `good`. Columns: predicted `bad`, predicted `good`)

* XGBoost:

```
[[ 35  25]
 [ 28 112]]
```

**Classification report highlights:**

* Random Forest is stronger in balancing both classes (macro F1 ≈ 0.71) and achieved higher ROC-AUC (0.77).
* XGBoost had slightly better recall for the `good` class, but Random Forest had better recall for the `bad` class and a better overall trade-off.

**Model selection rationale:**
For credit risk, **false negatives** (classifying a truly *bad* borrower as *good*) are costly. Random Forest showed a better balance between detecting `bad` applicants and correctly approving `good` applicants, had a higher ROC-AUC, and offered stable performance. Therefore, Random Forest was chosen as the production model and saved as `random_forest_credit_model.pkl`.

---

## Streamlit App (GUI)

A lightweight Streamlit app `app.py` was built so stakeholders can interact with the model.

Features of the app:

* Inputs: `Age`, `Sex`, `Job` (0–3), `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`.
* Encoders (`Sex_encoder.pkl`, `Housing_encoder.pkl`, etc.) are loaded at startup to transform inputs consistently.
* The app displays a preview of the encoded input, model prediction, and model confidence (if `predict_proba` is available).

**Run the app locally:**

```bash
streamlit run app.py
```

Place `random_forest_credit_model.pkl` and the encoder `.pkl` files in the same working directory as `app.py`.

---



## Results Summary (Quick Reference)

* Data: 1000 rows (700 good / 300 bad)
* Test split: 20% (200 samples)
* Final model: Random Forest (grid-searched)

  * **Accuracy:** 0.735
  * **ROC-AUC:** 0.7714
  * **Confusion matrix (test):** `[[43, 17], [36, 104]]`

Other models:

* XGBoost — Accuracy: 0.735, ROC-AUC: 0.7525, Confusion matrix: `[[35, 25], [28, 112]]`
* Decision Tree — Accuracy: 0.665
* ExtraTrees — Accuracy: 0.715

---

## Limitations & Future Work

**Limitations:**

* The project uses LabelEncoding for categorical features; depending on downstream models, One-Hot or target encoding may be preferable.
* The dataset is relatively small (1000 samples). More data or external features (e.g., credit history length, income) would improve model quality.
* Cost-sensitive evaluation (loss matrix) was not implemented; in production this should be prioritized.

**Future improvements:**

* Use **SHAP** or **LIME** for local and global explainability (required for regulatory/interpretability needs).
* Calibrate probabilities (Platt scaling / isotonic) to improve probability estimates.
* Add a REST API (FastAPI) wrapper to serve the model for systems integration.
* Run time-series/temporal validation if additional chronological data is available.
* Implement a full scikit-learn `Pipeline` with `ColumnTransformer` for robust preprocessing and avoid leakage.

**Thanks for checking out this project!**
