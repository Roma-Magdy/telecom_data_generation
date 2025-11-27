# telecom_data_generation
## All Batch Mode
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


## Batch and Streaming mode

### 1. Start Required Services with Docker Compose

From the project root, run:

```powershell
docker-compose up -d
```

This will start all required services (e.g., Kafka, Zookeeper, etc.) in the background.

### 2. Run Batch Data Generation

Generate the initial batch data by running:

```powershell
python data_generation/src/main_batch_only.py
```

This will create the initial batch of synthetic data in the appropriate output directories.

### 3. Run Streaming Data Generator

After the batch process finishes, start the streaming data generator:

```powershell
python stream_data_generation/stream_generator.py
```

This will begin generating and streaming new data (e.g., to Kafka) as configured.

---
You can stop the Docker services when finished with:

```powershell
docker-compose down
```
