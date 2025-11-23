"""
Invoice generator for postpaid subscriptions only.
- Generates a series of monthly invoices per postpaid subscription.
- Uses plan fees as base amount and applies TAX_RATE from config.
- Invoice status includes paid/unpaid/overdue.


Notes:
- Prepaid customers do not receive invoices in this simplification.
- You can extend this to bill for usage beyond allowances (overage charges) if required.
"""



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src import config
from src import utils

def generate_invoices(subscriptions: pd.DataFrame, plans: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plan_map = plans.set_index("plan_id").to_dict(orient="index")

    # choose subscriptions that are postpaid
    postpaid = subscriptions[subscriptions.subscription_type == "postpaid"]


    for _, sub in postpaid.iterrows():
        # find associated plan in subscription_plans? We assume 1:1 for simplicity
        # If multiple, adapt here to pick most recent.
        # For simplicity, sample a plan fee from plan catalog
        plan_row = plans.sample(1).iloc[0]
        plan_fee = plan_row["monthly_fee"]


        # number of invoices to generate: random (e.g., 3 to 24 months)
        months = np.random.randint(3, 25)
        for m in range(months):
            issue_date = datetime.now() - timedelta(days=30 * (m + 1))
            period_start = issue_date - timedelta(days=30)
            total = utils.money_round(plan_fee * (1 + config.TAX_RATE))


            rows.append({
            "invoice_id": utils.gen_uuid(),
            "customer_id": sub["customer_id"],
            "billing_period_start_date": period_start.date(),
            "billing_period_end_date": issue_date.date(),
            "issue_date": issue_date.date(),
            "allowance_days": np.random.choice([7, 14, 21]),
            "total_amount": total,
            "taxes": config.TAX_RATE,
            "status": np.random.choice(["paid", "paid", "paid", "unpaid", "overdue"]),
            "method": np.random.choice(["card", "cash", "wallet", "bank"])
            })


    return pd.DataFrame(rows)