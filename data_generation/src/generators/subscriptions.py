"""
Subscriptions generator.
Produces the `subscriptions` table with fields:
- subscription_id (PK), customer_id (FK), subscription_type, subscription_start_date, subscription_end_date


Design notes:
- A customer can have 1 or more subscriptions (e.g., secondary SIMs). We default
to 1 with a small chance of 2.
- subscription_type is determined by customer_segment and global ratios.
- subscription_end_date is None by default; churn simulation sets this when enabled.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src import config
from src import utils

def choose_subscription_type(customer_segment: str) -> str:
    """Return subscribed type using both segment bias and global ratios.


    VIP and corporate customers skew towards postpaid; normal mass customers skew prepaid.
    """
    r = np.random.random()
    if customer_segment in ["vip", "corporate"]:
        # 70% postpaid for VIP/corporate, 30% prepaid/hybrid mix
        if r < 0.70:
            return "postpaid"
        elif r < 0.99:
            return "prepaid"
        else:
            return "hybrid"
    else:
        # Normal customers follow global ratios (prepaid heavy)
        if r < config.PREPAID_RATIO:
            return "prepaid"
        elif r < config.PREPAID_RATIO + config.POSTPAID_RATIO:
            return "postpaid"
        else:
            return "hybrid"

def generate_subscriptions(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    now = datetime.now()


    for idx, cust in customers.iterrows():
        # most customers: 1 subscription; a small fraction have 2
        n_subs = np.random.choice([1, 2], p=[0.92, 0.08])
        for _ in range(n_subs):
            start_date = utils.random_date_between(
            datetime.combine(cust["signup_date"], datetime.min.time()),
            now
            ).date()


            rows.append({
            "subscription_id": utils.gen_uuid(),
            "customer_id": cust["customer_id"],
            "subscription_type": choose_subscription_type(cust["customer_segment"]),
            "subscription_start_date": start_date,
            "subscription_end_date": None
            })


    return pd.DataFrame(rows)