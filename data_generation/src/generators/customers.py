"""
Customers generator.
Produces the `customers` table with fields:
- customer_id (PK), first_name, last_name, date_of_birth, gender, signup_date,
status (active=1/churned=0), primary_phone, customer_segment, city, created_at


Design considerations (explained inline):
- We sample age using the Egypt distribution but enforce minimum age = 8.
- signup_date is uniform over a recent window (e.g., last 5 years)
- status default = 1 (active). If APPLY_CHURN is False, we will keep status=1.
- PII is synthetic (Faker) to avoid sensitive data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from src import config
from src import utils


fake = Faker()

def generate_customers(n_customers: int) -> pd.DataFrame:
    """
    Generate `n_customers` customer records.


    Parameters
    ----------
    n_customers : int
    Number of customers to generate.


    Returns
    -------
    pd.DataFrame
    customers table
    """
    rows = []


    # Sampling for segments uses the distribution defined in config
    seg_names = list(config.SEGMENT_DISTRIBUTION.keys())
    seg_weights = list(config.SEGMENT_DISTRIBUTION.values())


    now = datetime.now()
    signup_earliest = now - timedelta(days=365 * 5) # last 5 years


    for i in range(n_customers):
        age = utils.sample_age_from_buckets()
        if age < 8:
            age = 8
        # Build date_of_birth from age (approximate, day precision not critical)
        dob = (now - timedelta(days=365 * age)).date()


        seg = np.random.choice(seg_names, p=seg_weights)


        # signup_date sampled uniformly across last 5 years
        signup_dt = utils.random_date_between(signup_earliest, now).date()


        rows.append({
        "customer_id": utils.gen_uuid(),
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "date_of_birth": dob,
        "gender": np.random.choice(["M", "F"]),
        "signup_date": signup_dt,
        "status": 1, # default active; churn applied at subscription level optionally
        "primary_phone": utils.gen_phone_number(),
        "customer_segment": seg,
        "city": fake.city(),
        "created_at": datetime.now()
        })


        # small progress logging for very large runs
        if (i + 1) % 100_000 == 0: 
            print(f" generated {i+1} customers")

    return pd.DataFrame(rows)