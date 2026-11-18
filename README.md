# freddie-mac-loan-risk
Predicting 90+ day mortgage delinquency from origination data on 50,000 Freddie Mac loans (2016), comparing logistic regression and gradient boosting.

**Tools:** Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter

---

## Top Default Risk Drivers
<img width="990" height="790" alt="chartted" src="https://github.com/user-attachments/assets/0364d117-23ac-470b-b21f-44a8000b4c5a" />


*Logistic regression coefficients for the 15 strongest features. Red = raises default risk, blue = lowers it. Features were scaled first, so bar lengths can be compared directly.*

---

## Overview
This project builds a model that predicts whether a mortgage will go **90+ days late within its first 3 years**, using only information known on the day the loan was made (credit score, loan-to-value, debt-to-income, etc.).

**What it covers:**
- Building a default flag (`ever_90dpd`) from ~3.4 million rows of monthly loan performance data
- Cleaning Freddie Mac's filler codes and deciding which loans and features to keep
- Comparing two models with 5-fold stratified cross-validation:
  1. **Model 1: Logistic Regression** (simple, interpretable baseline)
  2. **Model 2: Gradient Boosting** (`HistGradientBoostingClassifier`)
- Scoring the final model once on a held-out test set and interpreting what drives risk

### Why I built this
I built this project to learn the **mortgage credit risk domain** and to get another full repetition of building a prediction model end to end, from raw data to a scored model. The notebook includes my own notes on how the data and the business work (zero balance codes, HARP loans, underwriting variables, how ROC-AUC and PR-AUC actually grade a model), written as I learned them.

---

## Key Results

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| 1 - Logistic Regression | 0.811 (± 0.034) | 0.070 (± 0.012) |
| 2 - Gradient Boosting | 0.816 (± 0.021) | 0.066 (± 0.014) |

The two models scored about the same (the differences are smaller than the ± spread across folds), so I chose **Logistic Regression** as the final model: it's simpler, it's the industry standard for credit scorecards, and every coefficient can be explained.

**Final model on the test set (scored once):**
- **ROC-AUC: 0.806.** In ~81% of pairs (one loan that defaulted, one that didn't), the defaulted loan got the higher predicted probability.
- **PR-AUC: 0.070.** About 7× better than random (baseline = the 1.0% default rate).
- Both landed close to the cross-validation estimates, so the model held up on loans it had never seen.

**Key findings:**
- **FICO is the strongest driver by far.** Default rate drops from 4.29% (620-679) to 0.28% (760+).
- **LTV and DTI** both raise risk, matching how lenders underwrite loans.
- **CLTV was dropped** because it almost duplicates LTV. With both in the model, LTV's coefficient flipped negative (the opposite of what the data shows). Removing CLTV fixed the sign without changing model performance.

---

## The Data
**Source:** [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), 2016 sample files

| File | Rows | What it is |
|---|---|---|
| Origination | 50,000 | One row per loan: borrower and loan details on the day it was made |
| Performance | ~3.4 million | One row per loan **per month**: whether the borrower was current or behind |

**Target (`ever_90dpd`):** a loan = 1 if, within its first 36 months, it was ever **90+ days delinquent** or ended in a **default-type exit** (foreclosure sale, short sale, or REO). Default rate: **~1.1%**.

---

## Code Highlights

**Loading pipe-delimited files with a separate header file**
```python
with open("data/origination_data_file_header.txt") as f:
    original_cols = f.read().strip().split("|")

orig = pd.read_csv(
    "data/sample_orig_2016.txt",
    sep="|",                 # pipe-delimited, not commas
    header=None,             # no header row in the file
    names=original_cols,     # use the names from the header file
    low_memory=False)
```

**Collapsing monthly rows to one row per loan (worst delinquency in 36 months)**
```python
perf_36 = perf[perf['LOAN AGE'] <= 36].copy()

loan_max = (
    perf_36.groupby('LOAN IDENTIFIER')
    .agg(max_months_dlq=('months_dlq', 'max'))
    .reset_index()
)
```

**Building the target from two checks**
```python
loan_max['ever_90dpd'] = (
    (loan_max['LOAN IDENTIFIER'].isin(zbc_default_ids)) |   # default-type Zero Balance Code
    (loan_max['max_months_dlq'] >= 3)                        # ever 90+ days late
).astype(int)
```

**Model 1: Logistic Regression with 5-fold stratified cross-validation**
```python
log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=1000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_reg_cv = cross_validate(
    log_reg, X_train, y_train,
    cv=cv,
    scoring=['roc_auc', 'average_precision']
)
```

**Model 2: Gradient Boosting on the same folds**
```python
hgb = HistGradientBoostingClassifier(
    class_weight='balanced',
    max_iter=300,          # number of trees
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

hgb_cv = cross_validate(hgb, X_train, y_train, cv=cv,
                        scoring=['roc_auc', 'average_precision'])
```

---

## Cleaning Decisions

| Decision | Why |
|---|---|
| Replaced DTI `999` with NaN | Freddie Mac filler code for "not available" |
| Dropped HARP loans (2,525) | Refinances of underwater borrowers: extreme LTVs (up to 336%), no DTI reported, a different population than normal originations |
| Dropped CLTV | Nearly identical to LTV, and caused LTV's coefficient to flip sign |
| Dropped IDs, dates, single-value columns, numeric location codes | No useful signal for the model |
| Filled `SPECIAL ELIGIBILITY PROGRAM` NaN with `'None'` | Blank means "not in a special program," not missing data |
| One-hot encoded 8 text columns | Models need numbers (`drop_first=True` to avoid multicollinearity) |

Final modeling data: **47,341 loans, 75 features**.

---

## Limitations
- **One year of loans (2016).** These loans were made during a strong housing market, so the default rate is low. The model hasn't been tested on a different year.
- **Probabilities aren't calibrated.** `class_weight='balanced'` pushes predicted probabilities up on purpose, so they're good for ranking loans but overstate the real chance of default.
- **No cutoff chosen.** ROC-AUC and PR-AUC grade how well loans are ranked. Turning that into approve/decline decisions would need a cutoff based on business costs.
- **State coefficients are noisy.** Many states have fewer than 200 loans, and all states are compared against Alaska (93 loans).

## Next Steps
- Choose a cutoff based on the cost of a missed default vs. a false alarm
- Out-of-time test: train on 2016 loans, test on a later year
- Tune Model 2 (tree depth, learning rate, number of trees)
- Calibrate the predicted probabilities

---

## Repository Structure
```
freddie-mac-loan-risk/
├── freddie_loan_prediction_model.ipynb   # full workflow: data prep, EDA, both models, results
├── images/
│   └── top15_risk_drivers.png            # chart used in this README
├── requirements.txt
└── README.md
```

**Data is not included in this repo.** Freddie Mac requires a free account to download the loan-level data. To run the notebook:
1. Register and download the 2016 sample files from the [Freddie Mac dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
2. Put `sample_orig_2016.txt`, `sample_perf_2016.txt`, and the two header files in a `data/` folder
3. Install the requirements and open the notebook:
```bash
pip install -r requirements.txt
jupyter notebook freddie_loan_prediction_model.ipynb
```

---

## AI Assistance
I used **Claude** (Anthropic) as a learning partner on this project: to explain mortgage domain concepts, check my reasoning, review/annotate my code, and help structure this README. I wrote the analysis and code myself and reviewed every explanation in the notebook.

---

## Contact
**Herman "Tre" McKinney III**
- LinkedIn: [linkedin.com/in/herman-mckinney-090b93214](https://www.linkedin.com/in/herman-mckinney-090b93214)
- GitHub: [github.com/hermanmckinney3](https://github.com/hermanmckinney3)
