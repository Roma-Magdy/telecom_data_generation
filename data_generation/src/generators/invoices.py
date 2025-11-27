# Fixed invoices.py
"""
Invoice generator for postpaid subscriptions - FIXED VERSION
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src import config
from src import utils


def generate_invoices(subscriptions: pd.DataFrame,
                      plans: pd.DataFrame,
                      subscription_plans: pd.DataFrame) -> pd.DataFrame:
    
    rows = []
    
    # Only postpaid subscriptions generate invoices
    postpaid = subscriptions[subscriptions.subscription_type == "postpaid"]
    
    # Debug: Check if we have subscription_plans data
    print(f"Debug - subscription_plans shape: {subscription_plans.shape}")
    print(f"Debug - subscription_plans columns: {subscription_plans.columns.tolist()}")
    
    # Get the latest plan for each subscription
    # Sort by subscription_id and start_date, then take the last entry per subscription
    latest_plan_map = (
        subscription_plans
        .sort_values(["subscription_id", "start_date"])
        .groupby("subscription_id")
        .last()[["plan_id"]]  # Take last plan_id
        .to_dict()["plan_id"]
    )
    
    # Create plan fee map
    plan_fee_map = plans.set_index("plan_id")["monthly_fee"].to_dict()
    
    # Debug: Check mappings
    print(f"Debug - Number of subscriptions with plans: {len(latest_plan_map)}")
    print(f"Debug - Number of plans with fees: {len(plan_fee_map)}")
    print(f"Debug - Sample plan fees: {dict(list(plan_fee_map.items())[:5])}")
    
    now = datetime.now()
    missing_plan_count = 0
    zero_fee_count = 0
    
    for _, sub in postpaid.iterrows():
        subscription_id = sub["subscription_id"]
        customer_id = sub["customer_id"]
        
        # Get the plan_id for this subscription
        plan_id = latest_plan_map.get(subscription_id)
        if plan_id is None:
            missing_plan_count += 1
            # Fallback: assign a default plan fee
            monthly_fee = 100.0  # Default fee
        else:
            # Get the monthly fee for this plan
            monthly_fee = plan_fee_map.get(plan_id, 0.0)
            if monthly_fee == 0.0:
                zero_fee_count += 1
                # Fallback: use a reasonable default based on subscription type
                monthly_fee = 100.0
        
        # Generate invoices for multiple months
        months = np.random.randint(3, 25)
        
        for m in range(months):
            invoice_date = now - timedelta(days=30 * (m + 1))
            billing_start = invoice_date - timedelta(days=30)
            billing_end = invoice_date
            due_date = invoice_date + timedelta(days=14)
            
            # Calculate total with tax
            total = utils.money_round(monthly_fee * (1 + config.TAX_RATE))
            
            # Payment behavior
            roll = np.random.random()
            
            if roll < 0.80:
                # 80% pay on time or slightly early
                delay = np.random.randint(-5, 6)
                payment_date = due_date + timedelta(days=delay)
                status = "paid"
            elif roll < 0.95:
                # 15% pay late
                delay = np.random.randint(1, 31)
                payment_date = due_date + timedelta(days=delay)
                status = "overdue"
            else:
                # 5% never pay
                payment_date = None
                delay = None
                status = "unpaid"
            
            rows.append({
                "invoice_id": utils.gen_uuid(),
                "subscription_id": subscription_id,
                "customer_id": customer_id,
                "billing_period_start_date": billing_start.date(),
                "billing_period_end_date": billing_end.date(),
                "invoice_date": invoice_date.date(),
                "due_date": due_date.date(),
                "payment_date": None if payment_date is None else payment_date.date(),
                "delay_days": delay if payment_date is not None else None,
                "total_amount": total,
                "tax_rate": config.TAX_RATE,
                "status": status,
                "method": np.random.choice(["card", "cash", "wallet", "bank"]),
            })
    
    print(f"Debug - Subscriptions missing plans: {missing_plan_count}")
    print(f"Debug - Plans with zero fees: {zero_fee_count}")
    
    return pd.DataFrame(rows)