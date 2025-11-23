"""
Utility functions used by generators.
"""
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import uuid
from typing import Tuple, List
from src import config


fake = Faker()
Faker_seed_applied = False

def seed_all(seed: int = None):
    """
    Seed all random generators for reproducibility.
    Call once at program start.
    """
    global Faker_seed_applied
    if seed is None:
        seed = config.RANDOM_SEED
    np.random.seed(seed)
    random.seed(seed)
    Faker.seed(seed)
    Faker_seed_applied = True

def sample_age_from_buckets() -> int:
    """
    Sample a single age using the AGE_BUCKETS distribution defined in config.
    Returns an integer age (years). The distribution is sampled at bucket level
    and then uniformly within a bucket.
    """
    buckets = config.AGE_BUCKETS
    ranges = [b[0] for b in buckets]
    weights = np.array([b[1] for b in buckets], dtype=float)
    weights /= weights.sum()


    chosen_idx = np.random.choice(len(ranges), p=weights)
    age_min, age_max = ranges[chosen_idx]
    # inclusive range sampling
    return int(np.random.randint(age_min, age_max + 1))

def random_date_between(start_date: datetime, end_date: datetime) -> datetime:
    """Return a random datetime between start_date and end_date."""
    if start_date >= end_date:
        return start_date
    delta = end_date - start_date
    int_delta = int(delta.total_seconds())
    random_second = random.randint(0, int_delta)
    return start_date + timedelta(seconds=random_second)

def gen_uuid() -> str:
    return str(uuid.uuid4())

def money_round(x: float) -> float:
    """Round currency amounts to 2 decimals for storage."""
    return float(round(x, 2))

def gen_phone_number() -> str:
    """Generate a plausible phone number string for Egypt-style numbers.
    This uses Faker as a fallback but can be customized to follow exact prefixes.
    """
    # e. g. Egyptian mobile prefixes: 010, 011, 012, 015 — but Faker may suffice
    prefix = 10
    suffix = random.randint(10_000_00, 99_999_99)
    return f"+20{prefix}{suffix}"

