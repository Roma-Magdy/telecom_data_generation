# fix_existing_files.py
"""
Fix subscription_plans to use only the 50 existing plan_ids
Then fix invoices to use correct fees
"""
import pandas as pd
import numpy as np
import glob

def fix_everything(batch_dir='./output_batch'):
    print("FIXING SUBSCRIPTION_PLANS AND INVOICES")
    print("="*60)
    
    # 1. Load the 50 existing plans
    plans = pd.read_parquet(f"{batch_dir}/plans.parquet")
    plan_ids = plans['plan_id'].tolist()
    plan_fee_map = plans.set_index("plan_id")["monthly_fee"].to_dict()
    
    print(f"Found {len(plans)} plans with fees from {plans['monthly_fee'].min()} to {plans['monthly_fee'].max()}")
    
    # 2. Fix subscription_plans - randomly assign from the 50 plans
    sp_files = glob.glob(f"{batch_dir}/subscription_plans_chunk*.parquet")
    
    for file in sp_files:
        sp_df = pd.read_parquet(file)
        # Replace all plan_ids with random selection from the 50 existing plans
        sp_df['plan_id'] = np.random.choice(plan_ids, size=len(sp_df))
        sp_df.to_parquet(file, index=False)
        print(f"Fixed {file}")
    
    # 3. Reload fixed subscription_plans and create mapping
    all_sp = []
    for file in sp_files:
        all_sp.append(pd.read_parquet(file))
    subscription_plans = pd.concat(all_sp, ignore_index=True)
    
    # Get latest plan for each subscription
    latest_plans = (
        subscription_plans
        .sort_values(['subscription_id', 'start_date'])
        .groupby('subscription_id')
        .last()
        .reset_index()
    )
    sub_to_plan_map = latest_plans.set_index('subscription_id')['plan_id'].to_dict()
    
    # 4. Fix invoices with correct fees
    invoice_files = glob.glob(f"{batch_dir}/invoices_chunk*.parquet")
    
    for file in invoice_files:
        invoices = pd.read_parquet(file)
        
        for idx, row in invoices.iterrows():
            sub_id = row['subscription_id']
            plan_id = sub_to_plan_map.get(sub_id)
            
            if plan_id and plan_id in plan_fee_map:
                monthly_fee = plan_fee_map[plan_id]
                total_amount = round(monthly_fee * 1.14, 2)  # Add 14% tax
                invoices.at[idx, 'total_amount'] = total_amount
        
        invoices.to_parquet(file, index=False)
        print(f"Fixed {file}")
    
    print("\n✅ ALL FILES FIXED!")
    
    # Verify
    test_invoice = pd.read_parquet(invoice_files[0])
    print(f"\nVerification - Unique invoice amounts: {test_invoice['total_amount'].nunique()}")
    print(f"Sample amounts: {test_invoice['total_amount'].value_counts().head()}")

if __name__ == "__main__":
    fix_everything('./output_batch')