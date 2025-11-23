# telecom_data_generation
This folder contains a modular, production-oriented synthetic data generator
for telecom CRM, billing, usage (CDR), and support tables suitable for
experimentation, ML model development (churn), and data pipeline testing.


Key features:
- Modular generators for each domain (customers, subscriptions, plans, invoices, topups, support, cdr)
- Fully configurable via `src/config.py`
- Churn simulation controlled by a flag (apply_churn = True/False)
- Scales to 1M+ customers
- Outputs Parquet files and small CSV samples
- Detailed code comments and explanations throughout


How to run:
1. Create a Python virtualenv and install requirements from requirements.txt
2. Edit `src/config.py` to set `N_CUSTOMERS` and `APPLY_CHURN` and other parameters
3. Run: `python src/main.py --output-dir ./output` or `python -m src.main`
