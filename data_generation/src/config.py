"""
Configuration: one place for controlling generation size, distributions, and flags.
Change values here or override via environment variables / CLI.
"""
from datetime import datetime


# Core generation size (set to 1_000_000 as requested)
N_CUSTOMERS = 1_000_000 # number of customer records to generate


# Churn control flag
APPLY_CHURN = True # True: simulate churn & subscription_end_date; False: skip churn


# Domain ratios and distributions
PREPAID_RATIO = 0.90
POSTPAID_RATIO = 0.09
HYBRID_RATIO = 0.01


SEGMENT_DISTRIBUTION = {
"normal": 0.90,
"vip": 0.05,
"corporate": 0.05
}

# Egypt age distribution for ages >= 8 (percent weights, not normalized)
AGE_BUCKETS = [
((8, 12), 17.8),
((13, 17), 9.4),
((18, 24), 11.6),
((25, 34), 15.3),
((35, 44), 13.8),
((45, 54), 9.8),
((55, 64), 6.7),
((65, 90), 5.0)
]


# Channel preferences for topups (41% offline, 41% online, 18% hybrid)
TOPUP_CHANNEL_DIST = {"online": 0.41, "offline": 0.41, "hybrid": 0.18}


# Plan price tiers in EGP (based on your guidance)
PLAN_PRICE_TIERS = {
"low": (20, 70),
"medium": (80, 150),
"high": (200, 350),
"vip": (500, 1500)
}


# ARPU anchors (used for guidance when building plan fees or checking totals)
ARPU_TELECOM_EGYPT = 2.6 # index / relative
ARPU_ETISALAT = 7.2

# Tax rate (used in invoice & topup tax calculations)
TAX_RATE = 0.14


# Base churn rate (annual) — your supplied statistic
BASE_ANNUAL_CHURN = 0.31


# Derived values: compute monthly churn probability exactly from annual
def monthly_churn_from_annual(annual):
    """Return monthly churn probability such that (1 - monthly)^12 = 1 - annual"""
    return 1 - (1 - annual) ** (1.0 / 12.0)


MONTHLY_CHURN = monthly_churn_from_annual(BASE_ANNUAL_CHURN)


# CDR & usage modeling parameters (tunable)
CDR_MEAN_EVENTS_PER_SUB_PER_MONTH = 20 # Poisson mean events per subscription per month (adjustable)

# Output options
DEFAULT_OUTPUT_DIR = "./output"
WRITE_PARQUET = True
WRITE_CSV_SAMPLE = True
CSV_SAMPLE_SIZE = 1000


# Random seed for reproducibility
RANDOM_SEED = 42