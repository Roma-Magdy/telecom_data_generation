"""
Orchestration script for the Telecom Synthetic Data Generator (chunked version).

Usage:
  python -m src.main --output-dir ./output --n-customers 1000000 --apply-churn --months-cdr 1
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

    # Ensure project root is in sys.path
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

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, n_customers - start_idx)
        print(f"\nGenerating chunk {chunk_idx+1}/{total_chunks}: customers {start_idx+1} to {start_idx+current_chunk_size}")

        # ----- CUSTOMERS -----
        customers_df = gen_customers.generate_customers(current_chunk_size)

        # ----- SUBSCRIPTIONS -----
        subscriptions_df = gen_subs.generate_subscriptions(customers_df)

        # ----- PLANS + SUBSCRIPTION_PLANS -----
        plans_df, subscription_plans_df = gen_plans.generate_plans_and_subscription_plans(
            subscriptions_df,
            customers_df
        )

        # ----- CHURN -----
        if apply_churn:
            subs = subscriptions_df.copy()
            monthly_p = config.MONTHLY_CHURN

            mask_active = subs.subscription_end_date.isna()
            random_vals = np.random.random(len(subs))
            churn_idx = subs.index[mask_active & (random_vals < monthly_p)]

            start_dates = pd.to_datetime(subs.loc[churn_idx, "subscription_start_date"])
            end_dates = [
                utils.random_date_between(sd.to_pydatetime(), pd.Timestamp.now().to_pydatetime())
                for sd in start_dates
            ]
            subs.loc[churn_idx, "subscription_end_date"] = pd.to_datetime(end_dates).date

            subscriptions_df = subs

            # Update customer status (1 active, 0 churned)
            active_counts = subscriptions_df[subscriptions_df.subscription_end_date.isna()] \
                                .groupby("customer_id").size()
            customers_df["status"] = 1
            churned = set(customers_df.customer_id) - set(active_counts.index)
            customers_df.loc[customers_df.customer_id.isin(list(churned)), "status"] = 0

        # ----- INVOICES -----
        # Pass subscription_plans_df to properly match plans to subscriptions
        invoices_df = gen_invoices.generate_invoices(
            subscriptions_df,
            plans_df,
            subscription_plans_df
        )

        # ----- TOPUPS -----
        topups_df = gen_topups.generate_topups(subscriptions_df)

        # ----- SUPPORT TICKETS -----
        tickets_df = gen_tickets.generate_support_tickets(subscriptions_df)

        # ----- CDR -----
        cdr_df = gen_cdr.generate_cdr(subscriptions_df, months=months_cdr)

        # ----- SAVE CHUNK -----
        def write_batch(df: pd.DataFrame, name: str):
            if df is None or len(df) == 0:
                print(f"[WARN] empty dataframe for {name}, skipping write.")
                return
            path = os.path.join(output_dir, f"{name}_chunk{chunk_idx+1}.parquet")
            df.to_parquet(path, index=False)
            print(f"Wrote {name} chunk -> {path} (rows={len(df)})")

        write_batch(customers_df, "customers")
        write_batch(subscriptions_df, "subscriptions")
        write_batch(plans_df, "plans")
        write_batch(subscription_plans_df, "subscription_plans")
        write_batch(invoices_df, "invoices")
        write_batch(topups_df, "topups")
        write_batch(tickets_df, "support_tickets")
        write_batch(cdr_df, "cdr")

    print("\nGeneration complete.")

    # ----- OPTIONAL CSV SAMPLES -----
    if config.WRITE_CSV_SAMPLE:
        sample_dir = os.path.join(output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        sample_count = config.CSV_SAMPLE_SIZE

        def safe_sample(df):
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return df.sample(min(sample_count, len(df)))

        safe_sample(customers_df).to_csv(os.path.join(sample_dir, "customers_sample.csv"), index=False)
        safe_sample(subscriptions_df).to_csv(os.path.join(sample_dir, "subscriptions_sample.csv"), index=False)
        safe_sample(plans_df).to_csv(os.path.join(sample_dir, "plans_sample.csv"), index=False)
        safe_sample(subscription_plans_df).to_csv(os.path.join(sample_dir, "subscription_plans_sample.csv"), index=False)
        safe_sample(invoices_df).to_csv(os.path.join(sample_dir, "invoices_sample.csv"), index=False)
        safe_sample(topups_df).to_csv(os.path.join(sample_dir, "topups_sample.csv"), index=False)
        safe_sample(tickets_df).to_csv(os.path.join(sample_dir, "support_tickets_sample.csv"), index=False)
        safe_sample(cdr_df).to_csv(os.path.join(sample_dir, "cdr_sample.csv"), index=False)

        print(f"Wrote CSV samples ({sample_count} rows max) → {sample_dir}")


if __name__ == "__main__":
    main()
