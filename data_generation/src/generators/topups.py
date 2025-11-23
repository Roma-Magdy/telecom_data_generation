"""
Topups generator for prepaid subscriptions.
- For each prepaid subscription, create a Poisson number of topup events.
- Amounts follow a log-normal distribution (to model small frequent recharges and occasional large ones).
- topup_channel follows TOPUP_CHANNEL_DIST.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src import config
from src import utils

def generate_topups(subscriptions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prepaid = subscriptions[subscriptions.subscription_type == "prepaid"]


    for _, sub in prepaid.iterrows():
        # events per subscription: Poisson (mean depends on business choice)
        n_topups = np.random.poisson(5)
        for _ in range(n_topups):
            amount = np.random.lognormal(mean=3.0, sigma=0.5)
            amount = utils.money_round(amount)
            tax_amount = utils.money_round(amount * config.TAX_RATE)
            channel = np.random.choice(list(config.TOPUP_CHANNEL_DIST.keys()),
            p=list(config.TOPUP_CHANNEL_DIST.values()))
            event_ts = utils.random_date_between(
            datetime.combine(sub["subscription_start_date"], datetime.min.time()),
            datetime.now()
            )
            rows.append({
            "topup_id": utils.gen_uuid(),
            "subscription_id": sub["subscription_id"],
            "amount": amount,
            "tax_amount": tax_amount,
            "topup_channel": channel,
            "timestamp": event_ts
            })
    return pd.DataFrame(rows)