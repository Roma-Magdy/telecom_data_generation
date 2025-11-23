"""
CDR generator (session-level records).
- Each subscription gets a Poisson number of events per month (configurable).
- call_duration_seconds ~ Log-normal (heavy tail) to model many short calls and few long ones.
- consumed_mb ~ Log-normal heavy-tailed to model web/app streaming spikes.
- Timestamp sampled over subscription active period.


Notes:
- Generating full-call-level CDRs for 1M subscriptions × many events will be huge; the module
is written to generate data in chunks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src import config
from src import utils

def generate_cdr(subscriptions: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """
    Generate session-level CDR events for the given subscriptions.


    Parameters
    ----------
    subscriptions : pd.DataFrame
    Subscriptions table
    months : int
    Number of months to simulate per subscription
    """
    rows = []
    events_mean = config.CDR_MEAN_EVENTS_PER_SUB_PER_MONTH * months


    for i, sub in subscriptions.iterrows():
        # number of events is Poisson-distributed
        n_events = np.random.poisson(events_mean)
        # cap events for memory-safety in single-machine runs (you can remove this cap in distributed mode)
        n_events = min(n_events, 5000)


        for _ in range(n_events):
            call_duration = max(1, int(np.random.lognormal(mean=2.0, sigma=1.0)))
            consumed_mb = float(np.random.lognormal(mean=1.5, sigma=1.2))
            ts = utils.random_date_between(
            datetime.combine(sub["subscription_start_date"], datetime.min.time()),
            datetime.now()
            )
            rows.append({
            "cdr_id": utils.gen_uuid(),
            "subscription_id": sub["subscription_id"],
            "call_duration_seconds": call_duration,
            "consumed_mb": round(consumed_mb, 3),
            "timestamp": ts
            })


        if (i + 1) % 10000 == 0:
            print(f" generated CDR for {i+1} subscriptions")


    return pd.DataFrame(rows)