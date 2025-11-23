"""
Orchestration script for the Telecom Synthetic Data Generator (chunked version).

Usage (from repo root):
  python -m src.main --output-dir ./output --n-customers 1000000 --apply-churn --months-cdr 1

Notes:
  - This version generates customers in chunks to avoid memory issues.
  - Each chunk is saved immediately to disk.
"""

import os
import sys
import math
import click
import numpy as np
import pandas as pd

from src import config
from src import utils
from src.generators import customers as gen_customers
from src.generators import subscriptions as gen_subs
from src.generators import plans as gen_plans
from src.generators import invoices as gen_invoices
from src.generators import topups as gen_topups
from src.generators import support_tickets as gen_tickets
from src.generators import cdr as gen_cdr


@click.command()
@click.option("--output-dir", default=config.DEFAULT_OUTPUT_DIR, help="Directory to write output files")
@click.option("--n-customers", default=config.N_CUSTOMERS, type=int, help="Number of customers to generate")
@click.option("--apply-churn/--no-churn", default=config.APPLY_CHURN, help="Whether to simulate churn")
@click.option("--months-cdr", default=1, type=int, help="Number of months to simulate CDR events per subscription")
@click.option("--chunk-size", default=100_000, type=int, help="Number of customers per chunk")
def main(output_dir, n_customers, apply_churn, months_cdr, chunk_size):
    # Make sure we run from project root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    utils.seed_all(config.RANDOM_SEED)
    config.N_CUSTOMERS = n_customers
    config.APPLY_CHURN = apply_churn
    os.makedirs(output_dir, exist_ok=True)

    print("Starting synthetic data generation (chunked):")
    print(f"  customers: {n_customers}")
    print(f"  apply_churn: {apply_churn}")
    print(f"  months_cdr: {months_cdr}")
    print(f"  chunk_size: {chunk_size}")
    print("--------------------------------------------------")

    total_chunks = math.ceil(n_customers / chunk_size)

    # Store batch file paths for optional concatenation later
    customer_files = []
    subscription_files = []
    plans_files = []
    subscription_plans_files = []
    invoices_files = []
    topups_files = []
    tickets_files = []
    cdr_files = []

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, n_customers - start_idx)
        print(f"\nGenerating chunk {chunk_idx + 1}/{total_chunks}: customers {start_idx + 1} to {start_idx + current_chunk_size}")

        # ---------- Customers ----------
        customers_df = gen_customers.generate_customers(current_chunk_size)

        # ---------- Subscriptions ----------
        subscriptions_df = gen_subs.generate_subscriptions(customers_df)

        # ---------- Plans ----------
        plans_df, subscription_plans_df = gen_plans.generate_plans_and_subscription_plans(subscriptions_df, customers_df)

        # ---------- Apply churn ----------
        if apply_churn:
            subs = subscriptions_df.copy()
            monthly_p = config.MONTHLY_CHURN

            # Vectorized churn simulation
            mask_active = subs.subscription_end_date.isna()
            random_vals = np.random.random(len(subs))
            churn_idx = subs.index[mask_active & (random_vals < monthly_p)]
            start_dates = pd.to_datetime(subs.loc[churn_idx, "subscription_start_date"])
            end_dates = [utils.random_date_between(sd.to_pydatetime(), pd.Timestamp.now().to_pydatetime()) for sd in start_dates]
            subs.loc[churn_idx, "subscription_end_date"] = pd.to_datetime(end_dates).date
            subscriptions_df = subs

            # Update customer status
            active_counts = subscriptions_df[subscriptions_df.subscription_end_date.isna()].groupby("customer_id").size()
            customers_df["status"] = 1
            churned_customers = set(customers_df.customer_id) - set(active_counts.index)
            customers_df.loc[customers_df.customer_id.isin(list(churned_customers)), "status"] = 0

        # ---------- Invoices ----------
        invoices_df = gen_invoices.generate_invoices(subscriptions_df, plans_df, customers_df)

        # ---------- Topups ----------
        topups_df = gen_topups.generate_topups(subscriptions_df)

        # ---------- Support Tickets ----------
        tickets_df = gen_tickets.generate_support_tickets(subscriptions_df)

        # ---------- CDR ----------
        cdr_df = gen_cdr.generate_cdr(subscriptions_df, months=months_cdr)

        # ---------- Write batch to disk ----------
        def write_batch(df: pd.DataFrame, name: str):
            if df is None or len(df) == 0:
                print(f"Warning: empty dataframe for {name}, skipping write.")
                return None
            batch_file = os.path.join(output_dir, f"{name}_chunk{chunk_idx + 1}.parquet")
            df.to_parquet(batch_file, index=False)
            print(f"Wrote {name} chunk -> {batch_file} (rows={len(df)})")
            return batch_file

        customer_files.append(write_batch(customers_df, "customers"))
        subscription_files.append(write_batch(subscriptions_df, "subscriptions"))
        plans_files.append(write_batch(plans_df, "plans"))
        subscription_plans_files.append(write_batch(subscription_plans_df, "subscription_plans"))
        invoices_files.append(write_batch(invoices_df, "invoices"))
        topups_files.append(write_batch(topups_df, "topups"))
        tickets_files.append(write_batch(tickets_df, "support_tickets"))
        cdr_files.append(write_batch(cdr_df, "cdr"))

    print("\nGeneration complete. All chunks written to disk.")

    # Optional CSV samples (from the last chunk only)
    if config.WRITE_CSV_SAMPLE:
        sample_dir = os.path.join(output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        sample_count = min(len(customers_df), config.CSV_SAMPLE_SIZE)
        customers_df.sample(sample_count).to_csv(os.path.join(sample_dir, "customers_sample.csv"), index=False)
        subscriptions_df.sample(sample_count).to_csv(os.path.join(sample_dir, "subscriptions_sample.csv"), index=False)
        plans_df.sample(sample_count).to_csv(os.path.join(sample_dir, "plans_sample.csv"), index=False)
        subscription_plans_df.sample(sample_count).to_csv(os.path.join(sample_dir, "subscription_plans_sample.csv"), index=False)
        invoices_df.sample(sample_count).to_csv(os.path.join(sample_dir, "invoices_sample.csv"), index=False)
        topups_df.sample(sample_count).to_csv(os.path.join(sample_dir, "topups_sample.csv"), index=False)
        tickets_df.sample(sample_count).to_csv(os.path.join(sample_dir, "support_tickets_sample.csv"), index=False)
        cdr_df.sample(sample_count).to_csv(os.path.join(sample_dir, "cdr_sample.csv"), index=False)
        print(f"Wrote CSV samples ({sample_count} rows each) in {sample_dir}")


if __name__ == "_main_":
    main()