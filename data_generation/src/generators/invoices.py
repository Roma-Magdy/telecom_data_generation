"""
Invoice generator for postpaid subscriptions.
Revised version with proper payment_date logic because payments table was removed.

KEY RULES:
----------
1) Only postpaid subscriptions produce invoices.
2) Each invoice includes:
      - invoice_date (date invoice was created)
      - due_date
      - payment_date
      - status (paid / overdue / unpaid)
      - delay_days (days between due_date and payment_date)
3) Payment logic:
      - paid: payment_date is before or after due_date
      - overdue: paid but payment_date > due_date
      - unpaid: no payment_date at all
4) Amount = plan_fee * (1 + TAX_RATE)
    (You may later add usage-based charges.)
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

    # Map subscription -> plan (latest plan)
    latest_plan_map = (
        subscription_plans
        .sort_values(["subscription_id", "start_date"])
        .groupby("subscription_id")
        .tail(1)[["subscription_id", "plan_id"]]
        .set_index("subscription_id")["plan_id"]
        .to_dict()
    )

    plan_fee_map = plans.set_index("plan_id")["monthly_fee"].to_dict()

    now = datetime.now()

    for _, sub in postpaid.iterrows():

        subscription_id = sub["subscription_id"]
        customer_id = sub["customer_id"]

        # Determine active plan fee
        plan_id = latest_plan_map.get(subscription_id)
        if plan_id is None:
            continue
        monthly_fee = plan_fee_map.get(plan_id, 0.0)

        # Base invoice count per subscription (random)
        months = np.random.randint(3, 25)

        # Starting point for backdated invoices
        for m in range(months):

            # Invoice happens monthly
            invoice_date = now - timedelta(days=30 * (m + 1))
            billing_start = invoice_date - timedelta(days=30)
            billing_end = invoice_date

            # Due date is 14 days after invoice
            due_date = invoice_date + timedelta(days=14)

            # Compute charge
            total = utils.money_round(monthly_fee * (1 + config.TAX_RATE))

            # Determine payment behavior
            roll = np.random.random()

            if roll < 0.80:
                # 80% pay on time or slightly early
                delay = np.random.randint(-5, 6)    # can be slightly before due_date
                payment_date = due_date + timedelta(days=delay)
                status = "paid"

            elif roll < 0.95:
                # 15% pay late
                delay = np.random.randint(1, 31)    # 1–30 days late
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

    return pd.DataFrame(rows)
