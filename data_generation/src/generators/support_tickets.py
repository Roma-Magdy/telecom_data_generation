"""
Support ticket generator.
- Tickets are relatively rare per subscription (Poisson low mean).
- Each ticket has created_at, resolved_at (may be None for open), status, agent_id, and priority.
- Unresolved or escalated tickets are useful signals for churn analytic models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src import utils

def generate_support_tickets(subscriptions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sub in subscriptions.iterrows():
        n_tickets = np.random.poisson(0.15)
        for _ in range(n_tickets):
            created_at = utils.random_date_between(
            datetime.combine(sub["subscription_start_date"], datetime.min.time()),
            datetime.now()
            )
            # 20% chance ticket remains unresolved
            if np.random.random() < 0.80:
                resolved_at = created_at + timedelta(days=np.random.randint(1, 30))
                status = np.random.choice(["closed", "closed", "closed", "escalated"]) # more closed than escalated
            else:
                resolved_at = None
                status = "open"


            rows.append({
            "ticket_id": utils.gen_uuid(),
            "customer_id": sub["customer_id"],
            "subscription_id": sub["subscription_id"],
            "category": np.random.choice(["billing", "network", "device", "sim"]),
            "created_at": created_at,
            "resolved_at": resolved_at,
            "status": status,
            "agent_id": np.random.randint(1000, 2000),
            "priority": np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1])
            })
    return pd.DataFrame(rows)