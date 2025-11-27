"""
Batch data generator - saves chunks immediately
"""
import os
import sys
import click
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import utils
from src.generators import customers as gen_customers
from src.generators import subscriptions as gen_subs
from src.generators import plans as gen_plans
from src.generators import invoices as gen_invoices

@click.command()
@click.option("--output-dir", default="./output_batch", help="Directory for batch files")
@click.option("--n-customers", default=1000000, type=int, help="Number of customers")
@click.option("--apply-churn", is_flag=True, help="Apply churn simulation")
@click.option("--chunk-size", default=50000, type=int, help="Smaller chunks for faster saves")
def main(output_dir, n_customers, apply_churn, chunk_size):
    
    utils.seed_all(config.RANDOM_SEED)
    config.N_CUSTOMERS = n_customers
    config.APPLY_CHURN = apply_churn
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Optimized Batch Generation")
    print(f"  customers: {n_customers:,}")
    print(f"  chunk_size: {chunk_size:,}")
    print(f"  output: {output_dir}")
    print("-" * 50)
    
    total_chunks = (n_customers + chunk_size - 1) // chunk_size
    start_time = datetime.now()
    
    # Generate plans once (global catalog)
    print("Generating global plans catalog...")
    dummy_customers = pd.DataFrame({'customer_id': ['dummy'], 'customer_segment': ['normal']})
    dummy_subs = pd.DataFrame({
        'subscription_id': ['dummy'],
        'customer_id': ['dummy'],
        'subscription_type': ['prepaid'],
        'subscription_start_date': [datetime.now().date()],
        'subscription_end_date': [None]
    })
    plans_df, _ = gen_plans.generate_plans_and_subscription_plans(dummy_subs, dummy_customers)
    plans_df.to_parquet(f"{output_dir}/plans.parquet", index=False)
    print(f"✓ Saved {len(plans_df)} plans")
    
    # Process chunks
    for chunk_idx in range(total_chunks):
        chunk_start = datetime.now()
        start_idx = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, n_customers - start_idx)
        
        print(f"\n[Chunk {chunk_idx+1}/{total_chunks}] Generating {current_chunk_size:,} customers...")
        
        # Generate this chunk
        customers_df = gen_customers.generate_customers(current_chunk_size)
        subscriptions_df = gen_subs.generate_subscriptions(customers_df)
        
        # Generate subscription_plans for this chunk only
        _, subscription_plans_df = gen_plans.generate_plans_and_subscription_plans(
            subscriptions_df, customers_df
        )
        
        # Apply churn if requested
        if apply_churn:
            subs = subscriptions_df.copy()
            monthly_p = config.MONTHLY_CHURN
            
            mask_active = subs.subscription_end_date.isna()
            random_vals = np.random.random(len(subs))
            churn_idx = subs.index[mask_active & (random_vals < monthly_p)]
            
            if len(churn_idx) > 0:
                start_dates = pd.to_datetime(subs.loc[churn_idx, "subscription_start_date"])
                end_dates = [
                    utils.random_date_between(sd.to_pydatetime(), pd.Timestamp.now().to_pydatetime())
                    for sd in start_dates
                ]
                subs.loc[churn_idx, "subscription_end_date"] = pd.to_datetime(end_dates).date
            
            subscriptions_df = subs
            
            # Update customer status
            active_counts = subscriptions_df[subscriptions_df.subscription_end_date.isna()] \
                                .groupby("customer_id").size()
            customers_df["status"] = 1
            churned = set(customers_df.customer_id) - set(active_counts.index)
            customers_df.loc[customers_df.customer_id.isin(list(churned)), "status"] = 0
        
        # Generate invoices (only for postpaid)
        postpaid_subs = subscriptions_df[subscriptions_df.subscription_type == 'postpaid']
        if len(postpaid_subs) > 0:
            invoices_df = gen_invoices.generate_invoices(
                subscriptions_df, plans_df, subscription_plans_df
            )
        else:
            invoices_df = pd.DataFrame()
        
        # Save this chunk immediately
        customers_df.to_parquet(f"{output_dir}/customers_chunk{chunk_idx+1}.parquet", index=False)
        subscriptions_df.to_parquet(f"{output_dir}/subscriptions_chunk{chunk_idx+1}.parquet", index=False)
        subscription_plans_df.to_parquet(f"{output_dir}/subscription_plans_chunk{chunk_idx+1}.parquet", index=False)
        
        if len(invoices_df) > 0:
            invoices_df.to_parquet(f"{output_dir}/invoices_chunk{chunk_idx+1}.parquet", index=False)
        
        chunk_time = (datetime.now() - chunk_start).total_seconds()
        print(f"  ✓ Chunk {chunk_idx+1} saved in {chunk_time:.1f}s")
        print(f"    - {len(customers_df):,} customers")
        print(f"    - {len(subscriptions_df):,} subscriptions")
        print(f"    - {len(invoices_df):,} invoices")
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*50}")
    print(f"✓ Batch generation complete in {total_time/60:.1f} minutes")
    print(f"✓ Files saved to: {output_dir}")
    print(f"✓ Total: {n_customers:,} customers in {total_chunks} chunks")

if __name__ == "__main__":
    main()