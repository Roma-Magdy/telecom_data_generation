"""
Plans generator and subscription_plans history.

- Each subscription gets its own plan (simplified model).
- In real telco systems, you may have a shared plan catalog and assign the same
  plan_id to many subscriptions.

Produces two tables:
1) plans (plan_id, category, monthly_fee, data_allowance_mb, voice_allowance_min)
2) subscription_plans (subscription_plan_id, subscription_id, plan_id, start_date, end_date)
"""

import pandas as pd
import numpy as np
import datetime  # <-- correct import

from src import config
from src import utils


def generate_monthly_fee_by_segment(segment: str) -> float:
    """
    Generate monthly fee based on customer segment and ARPU-derived tiers.
    - VIP/corporate → higher fees.
    - Normal → sample across low/medium/high with weighted probability.
    """
    tiers = config.PLAN_PRICE_TIERS

    if segment == "vip":
        return float(np.random.randint(*tiers["vip"]))

    if segment == "corporate":
        return float(np.random.randint(*tiers["high"]))

    # normal users: biased to low/medium tiers
    pick = np.random.random()
    if pick < 0.6:
        return float(np.random.randint(*tiers["low"]))
    elif pick < 0.95:
        return float(np.random.randint(*tiers["medium"]))
    else:
        return float(np.random.randint(*tiers["high"]))


def generate_data_allowance_by_age(age: int) -> int:
    """Data allowance rules:
    - 18–25 → heavy data users: very high MB.
    - <17 → children: low MB.
    - >40 → moderate MB.
    - otherwise → mid-range MB.
    """
    if 18 <= age <= 25:
        return int(np.random.randint(12_000, 20_000))
    if age <= 17:
        return int(np.random.randint(1_000, 3_000))
    if age > 40:
        return int(np.random.randint(2_000, 8_000))

    return int(np.random.randint(4_000, 12_000))


def generate_voice_allowance_by_age(age: int) -> int:
    """Voice allowance increases slightly for older users."""
    if age > 40:
        return int(np.random.randint(200, 400))
    return int(np.random.randint(50, 200))


def generate_plans_and_subscription_plans(subscriptions: pd.DataFrame, customers: pd.DataFrame):
    """
    Create plans + subscription plan history.
    For each subscription:
      - derive customer age
      - generate plan characteristics (fee, data, voice)
      - attach the plan to subscription

    Returns:
        plans_df, subplans_df
    """

    plans_rows = []
    subplan_rows = []

    # Map customer_id → customer row for fast lookup
    cust_map = customers.set_index("customer_id").to_dict(orient="index")

    for _, sub in subscriptions.iterrows():

        cust = cust_map[sub["customer_id"]]

      
        age = int((datetime.datetime.now().date() - cust["date_of_birth"]).days / 365)

        # Plan generation rules
        monthly_fee = generate_monthly_fee_by_segment(cust["customer_segment"])
        data_mb = generate_data_allowance_by_age(age)
        voice_min = generate_voice_allowance_by_age(age)

        # Create plan
        plan_id = utils.gen_uuid()
        plans_rows.append({
            "plan_id": plan_id,
            "category": "mixed",
            "monthly_fee": utils.money_round(monthly_fee),
            "data_allowance_mb": data_mb,
            "voice_allowance_min": voice_min
        })

        # Create subscription–plan link
        subplan_rows.append({
            "subscription_plan_id": utils.gen_uuid(),
            "subscription_id": sub["subscription_id"],
            "plan_id": plan_id,
            "start_date": sub["subscription_start_date"],
            "end_date": None
        })

    plans_df = pd.DataFrame(plans_rows).drop_duplicates(subset=["plan_id"]).reset_index(drop=True)
    subplans_df = pd.DataFrame(subplan_rows)

    return plans_df, subplans_df
