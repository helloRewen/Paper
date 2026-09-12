# Experiment Code and Data

This package contains the data-processing procedures, model-estimation code, statistical analyses, and tabular outputs used in the study of prefix-based early warning and process governance. Each manuscript table has one executable Python file with the same table number. Figure-generation code and graphical outputs are not included.

## Data

| Dataset ID                       | Dataset name                                          | File                                               |
| -------------------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| `corporate_account_opening`      | Corporate Account-opening Event Log                   | `data/private/corporate_account_opening.csv`       |
| `small_business_credit_approval` | Small-business Credit Approval Event Log              | `data/private/small_business_credit_approval.xlsx` |
| `bpi2014_rabobank_ict`           | BPI Challenge 2014 Rabobank ICT Incident Activity Log | `data/public/bpi2014_rabobank_ict.csv`             |
| `bpi2017_offer_log`              | BPI Challenge 2017 Offer Log                          | `data/public/bpi2017_offer_log.xes.gz`             |
| `bpi2020_request_for_payment`    | BPI Challenge 2020 Request for Payment Log            | `data/public/bpi2020_request_for_payment.xes.gz`   |

The two bank-owned datasets are not redistributed. Place authorized copies under `data/private` before running the complete workflow. The three BPI Challenge logs are stored under `data/public`.

## Code Layout

```text
scripts/
  run_all_tables.py
  table_01_dataset_statistics.py
  table_02_operational_actionability.py
  table_03_feature_ablation_performance.py
  table_04_signal_marginal_contribution.py
  table_05_deviation_type_performance.py
  table_06_pgv_evaluation.py
  table_07_variant_signal_correlation.py
  table_a1_cross_context_performance.py
  table_a2_cross_context_pgv.py
  verify_package.py
src/
  data loading, feature construction, model estimation, and statistical analysis
results/paper_tables/
  one final CSV for each manuscript table
```

The files under `scripts` are the reader-facing entry points. The files under `src` contain shared implementations used by more than one table and are not separate experiments.

## Environment

The recorded environment is Python 3.12.10.

```
numpy==2.4.4
openpyxl==3.1.5
pandas==3.0.2
scikit-learn==1.8.0
scipy==1.17.1
torch==2.12.1
transformers==4.57.6
xgboost==3.2.0
```

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Execution

Generate all tables:

```powershell
.venv\Scripts\python scripts\run_all_tables.py
```
