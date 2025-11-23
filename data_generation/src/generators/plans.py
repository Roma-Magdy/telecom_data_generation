"""
src/generators/plans.py

Global market plan catalog + subscription_plan history generator
-----------------------------------------------------------------

Features implemented (explicit):

1) Global plan catalog (Option C1: large catalog, ~35-50 plans)
   - Prepaid, Postpaid, Corporate, Youth, Senior, Add-on categories
   - Each plan has: plan_id, plan_name, category, monthly_fee,
     data_allowance_mb, voice_allowance_min, allowed_segments, description

2) Assign subscriptions to plans (no 1:1 plan-per-subscription)
   - Choose initial plan consistent with subscription_type and customer_segment
   - Use a bias to prefer lower-fee plans more often (skew)

3) Simulate plan-change history for each subscription
   - Plan-change frequency distribution (exactly as requested):
       60% -> 0 changes
       30% -> 1 change
       10% -> 2-4 changes  (distribution inside this 10% allocated as 6/3/1)
   - When a change occurs, previous plan's end_date = next plan's start_date
   - Final plan end_date = NULL for active subscriptions

4) Churn rule (Rule A)
   - If subscription has a non-null subscription_end_date (churn), then
     the final subscription_plan row's end_date is set to subscription_end_date.

5) Date range rule (D1)
   - Plan start dates are sampled within the last 3 years (relative to now),
     but they will never be earlier than the subscription_start_date.
     This produces realistic plan histories while respecting subscription lifetime.

Performance & notes:
- This implementation loops subscriptions in Python. It scales to millions but
  for extremely large workloads consider batching/streaming and writing to Parquet
  in chunks (Dask / PySpark recommended).
- The code is defensive: it handles missing/invalid dates and falls back to reasonable defaults.
"""

from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import datetime

from src import config
from src import utils

# Seed numpy RNG for reproducibility inside this module (main should call utils.seed_all())
np.random.seed(config.RANDOM_SEED)


# ---------------------------
# Helper: build a catalog plan row
# ---------------------------
def _make_plan_row(plan_name: str,
                   category: str,
                   monthly_fee: float,
                   data_mb: int,
                   voice_min: int,
                   allowed_segments: List[str],
                   description: str = "") -> Dict:
    """
    Create a single plan entry as a dict for the catalog.
    """
    return {
        "plan_id": utils.gen_uuid(),
        "plan_name": plan_name,
        "category": category,
        "monthly_fee": utils.money_round(monthly_fee),
        "data_allowance_mb": int(data_mb),
        "voice_allowance_min": int(voice_min),
        "allowed_segments": ",".join(allowed_segments),
        "description": description
    }


# ---------------------------
# Generate large plan catalog (C1)
# ---------------------------
def generate_plan_catalog(prepaid_count=15,
                          postpaid_count=15,
                          corporate_count=7,
                          youth_count=5,
                          senior_count=3,
                          addon_count=5) -> pd.DataFrame:
    """
    Create a large, realistic plan catalog.

    Parameters allow control of catalog size; defaults tuned for Option C1.
    """
    plans = []

    # Prepaid plans: small -> medium -> large
    for i in range(prepaid_count):
        base = 20 + i * 8
        fee = float(np.random.randint(max(10, int(base - 10)), int(base + 60)))
        # data scales with index: small plans ~ hundreds MB, big ones many GB
        data_mb = int(np.random.randint(200, 2000) * (1 + i * 0.6))
        voice_min = int(np.random.randint(20, 200) * (1 + i * 0.15))
        name = f"Prepaid-{i+1}"
        plans.append(_make_plan_row(name, "prepaid", fee, data_mb, voice_min, ["normal", "youth"]))

    # Postpaid plans: tiered, higher base fees & allowances
    for i in range(postpaid_count):
        base = 80 + i * 30
        fee = float(np.random.randint(max(50, int(base - 30)), int(base + 300)))
        data_mb = int(np.random.randint(3_000, 10_000) * (1 + i * 0.4))
        voice_min = int(np.random.randint(100, 400) * (1 + i * 0.25))
        name = f"Postpaid-{i+1}"
        plans.append(_make_plan_row(name, "postpaid", fee, data_mb, voice_min, ["normal", "vip", "corporate"]))

    # Corporate plans: much higher allowances
    for i in range(corporate_count):
        fee = float(np.random.randint(400, 3000))
        data_mb = int(np.random.randint(20_000, 200_000))
        voice_min = int(np.random.randint(500, 5000))
        name = f"Corporate-{i+1}"
        plans.append(_make_plan_row(name, "corporate", fee, data_mb, voice_min, ["corporate", "vip"]))

    # Youth plans: social/gaming heavy
    for i in range(youth_count):
        fee = float(np.random.randint(30, 300))
        data_mb = int(np.random.randint(10_000, 80_000))
        voice_min = int(np.random.randint(30, 200))
        name = f"Youth-{i+1}"
        plans.append(_make_plan_row(name, "youth", fee, data_mb, voice_min, ["youth", "normal"], "Social/Gaming focused"))

    # Senior / low usage plans
    for i in range(senior_count):
        fee = float(np.random.randint(15, 80))
        data_mb = int(np.random.randint(100, 2_000))
        voice_min = int(np.random.randint(100, 600))
        name = f"Senior-{i+1}"
        plans.append(_make_plan_row(name, "senior", fee, data_mb, voice_min, ["normal", "senior"], "Voice-oriented"))

    # Addons / boosters
    for i in range(addon_count):
        fee = float(np.round(np.random.uniform(5.0, 80.0), 2))
        data_mb = int(np.random.randint(500, 50_000))
        voice_min = int(np.random.randint(10, 500))
        name = f"Addon-{i+1}"
        plans.append(_make_plan_row(name, "addon", fee, data_mb, voice_min, ["normal", "vip", "corporate"], "Booster/Add-on"))

    plans_df = pd.DataFrame(plans)
    # Ensure unique
    plans_df = plans_df.drop_duplicates(subset=["plan_id"]).reset_index(drop=True)
    return plans_df


# ---------------------------
# Plan selection helper
# ---------------------------
def _choose_plan_for_subscription(sub_row: pd.Series, plans_df: pd.DataFrame) -> pd.Series:
    """
    Choose a single plan from plans_df for the given subscription row.

    Priority/filtering logic:
      - match subscription_type (prepaid/postpaid/hybrid)
      - bias by customer_segment when available (prefer plans that list the segment)
      - weight sampling so lower-fee plans are more common (skew towards cheaper options)
    """
    sub_type = sub_row.get("subscription_type", None)
    cust_segment = sub_row.get("customer_segment", None)

    # Filter by category
    if sub_type == "prepaid":
        candidates = plans_df[plans_df["category"] == "prepaid"].copy()
    elif sub_type == "postpaid":
        candidates = plans_df[plans_df["category"] == "postpaid"].copy()
    else:
        # hybrid or unknown: allow prepaid/postpaid
        candidates = plans_df[plans_df["category"].isin(["prepaid", "postpaid"])].copy()

    # If we have customer_segment, prefer plans that mention it in allowed_segments
    if cust_segment is not None:
        mask = candidates["allowed_segments"].str.contains(cust_segment)
        if mask.any():
            candidates = candidates[mask]

    # Fallback to entire catalog if no candidates remain
    if candidates.empty:
        candidates = plans_df.copy()

    # Weighted by inverse fee (cheaper plans more probable)
    fees = candidates["monthly_fee"].astype(float).fillna(0.0)
    inv = 1.0 / (fees + 1.0)  # avoid division by zero
    probs = inv / inv.sum()

    chosen = candidates.sample(weights=probs, replace=True).iloc[0]
    return chosen


# ---------------------------
# Main entrypoint: build plans + subscription_plans history
# ---------------------------
def generate_plans_and_subscription_plans(subscriptions: pd.DataFrame,
                                          customers: pd.DataFrame,
                                          catalog_args: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build global catalog and realistic subscription_plan history.

    Parameters
    ----------
    subscriptions : pd.DataFrame
        subscriptions table; expected columns: subscription_id, customer_id,
        subscription_type, subscription_start_date, subscription_end_date (nullable)
    customers : pd.DataFrame
        customers table; expected columns include customer_id, customer_segment, date_of_birth
    catalog_args : dict, optional
        to override counts for catalog generation

    Returns
    -------
    plans_df : pd.DataFrame
        global plan catalog
    subscription_plans_df : pd.DataFrame
        rows: subscription_plan_id, subscription_id, plan_id, start_date, end_date
    """
    # default catalog sizes tuned for Option C1 (total ~50)
    if catalog_args is None:
        catalog_args = {
            "prepaid_count": 15,
            "postpaid_count": 15,
            "corporate_count": 7,
            "youth_count": 5,
            "senior_count": 3,
            "addon_count": 5
        }

    plans_df = generate_plan_catalog(**catalog_args)

    # Build a quick customer lookup map (for segment & DOB access)
    cust_map = customers.set_index("customer_id").to_dict(orient="index")

    subplan_rows = []

    now_dt = datetime.datetime.now()
    three_years_ago = now_dt - datetime.timedelta(days=365 * 3)

    # Plan-change probabilites matching your specified Rule B:
    # 60% zero changes, 30% one change, 10% two-to-four changes
    # We'll break the 10% into 6% (2 changes), 3% (3 changes), 1% (4 changes)
    change_choices = [0, 1, 2, 3, 4]
    change_probs = [0.60, 0.30, 0.06, 0.03, 0.01]

    # Iterate over subscriptions. For extremely large data sets consider chunking.
    for _, sub in subscriptions.iterrows():
        subscription_id = sub["subscription_id"]
        customer_id = sub["customer_id"]

        # subscription_start_date from subscriptions table
        # fallback to three years ago if missing/invalid
        try:
            sub_start = pd.to_datetime(sub["subscription_start_date"]).to_pydatetime()
            # ensure sub_start is not in the future; if so, clamp to now
            if sub_start > now_dt:
                sub_start = now_dt - datetime.timedelta(days=np.random.randint(0, 30))
        except Exception:
            sub_start = three_years_ago

        # enforce D1: plan history should start no earlier than (now - 3 years)
        effective_earliest = max(sub_start, three_years_ago)

        # subscription_end_date (churn marker) — could be None
        sub_end_raw = sub.get("subscription_end_date", None)
        if pd.isna(sub_end_raw) or sub_end_raw is None:
            subscription_end_date = None
        else:
            try:
                subscription_end_date = pd.to_datetime(sub_end_raw).date()
            except Exception:
                subscription_end_date = pd.to_datetime(sub_end_raw, errors="coerce").date()

        # customer_segment (if available)
        cust_info = cust_map.get(customer_id, {})
        cust_segment = cust_info.get("customer_segment", None)

        # Build a temporary sub_row used by plan chooser
        tmp_subrow = {
            "subscription_id": subscription_id,
            "subscription_type": sub.get("subscription_type", None),
            "customer_segment": cust_segment
        }

        # Choose initial plan
        initial_plan = _choose_plan_for_subscription(tmp_subrow, plans_df)

        # Decide number of plan changes using the distribution above
        n_changes = np.random.choice(change_choices, p=change_probs)

        # If subscription has very short lifetime we might force n_changes=0
        if subscription_end_date is not None:
            # compute lifetime in days
            try:
                lifetime_days = (pd.to_datetime(subscription_end_date) - pd.to_datetime(effective_earliest)).days
                if lifetime_days < 30:
                    # too short to realistically change plan
                    n_changes = 0
            except Exception:
                pass

        # Build plan start timestamps list
        plan_start_times = []

        # initial plan start: choose between effective_earliest and now (but not in future)
        # We'll randomize within [effective_earliest, now_dt - 7 days] to leave room for changes
        latest_for_start = now_dt - datetime.timedelta(days=7)
        if effective_earliest >= latest_for_start:
            # subscription started very recently; use effective_earliest
            plan_start_times.append(effective_earliest)
        else:
            # randomize initial start uniformly between effective_earliest and latest_for_start
            span_seconds = int((latest_for_start - effective_earliest).total_seconds())
            offset = np.random.randint(0, max(1, span_seconds))
            plan_start_times.append(effective_earliest + datetime.timedelta(seconds=int(offset)))

        # If there are changes, sample change times between the initial start and now - 7 days
        if n_changes > 0:
            lower = plan_start_times[0] + datetime.timedelta(days=7)
            upper = now_dt - datetime.timedelta(days=7)
            if lower >= upper:
                # Not enough time to schedule changes; treat as 0 changes
                n_changes = 0
            else:
                # sample n_changes distinct times and sort ascending
                random_seconds = np.random.randint(0, int((upper - lower).total_seconds()), size=n_changes)
                random_dates = [lower + datetime.timedelta(seconds=int(s)) for s in random_seconds]
                random_dates_sorted = sorted(random_dates)
                plan_start_times.extend(random_dates_sorted)

        # For each planned start time, choose a plan and create subscription_plan row (end_date set later)
        last_plan_id = None
        for pst in plan_start_times:
            # make a selection; we temporarily pass sub metadata to chooser
            tmp_subrow["subscription_type"] = sub.get("subscription_type", None)
            chosen_plan = _choose_plan_for_subscription(tmp_subrow, plans_df)

            # small heuristic: avoid same plan repeating immediately; allow it sometimes
            if chosen_plan["plan_id"] == last_plan_id:
                if np.random.random() < 0.7:
                    # accept same plan (no effective change) — keep it (we'll still write it)
                    pass
                else:
                    # try to re-pick a different plan from same category (best-effort)
                    candidates = plans_df[plans_df["category"] == chosen_plan["category"]]
                    if len(candidates) > 1:
                        alt = candidates[candidates["plan_id"] != chosen_plan["plan_id"]]
                        if len(alt) > 0:
                            chosen_plan = alt.sample(1).iloc[0]

            subplan_rows.append({
                "subscription_plan_id": utils.gen_uuid(),
                "subscription_id": subscription_id,
                "plan_id": chosen_plan["plan_id"],
                "start_date": pd.to_datetime(pst).date(),
                "end_date": None  # will be fixed in the pass below
            })
            last_plan_id = chosen_plan["plan_id"]

        # NOTE: we intentionally do not create an extra "final plan" beyond initial+changes
        # the final plan (last in subplan_rows per subscription) will remain active (end_date None)
        # unless the subscription has subscription_end_date (churn), in which case we set it below.

    # Build DataFrame from generated rows
    subplans_df = pd.DataFrame(subplan_rows)

    # If no rows, return empty
    if subplans_df.empty:
        return plans_df, subplans_df

    # Sort and set previous end_date = next start_date for each subscription
    subplans_df.sort_values(["subscription_id", "start_date"], inplace=True, ignore_index=True)

    # Group and fill end_date for previous rows
    def _assign_sequential_end_dates(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("start_date").reset_index(drop=True)
        for i in range(len(group) - 1):
            next_start = group.at[i + 1, "start_date"]
            # previous plan ends exactly when next begins
            group.at[i, "end_date"] = pd.to_datetime(next_start).date()
        # final plan end_date remains None for now (handled by churn rule next)
        return group

    subplans_df = subplans_df.groupby("subscription_id", group_keys=False).apply(_assign_sequential_end_dates)
    subplans_df.reset_index(drop=True, inplace=True)

    # Now apply churn rule (Rule A): if subscription has subscription_end_date, set final plan end_date to that date
    # Build map subscription_id -> subscription_end_date for fast lookup
    subs_end_map = subscriptions.set_index("subscription_id")["subscription_end_date"].to_dict()

    # For each subscription group, set the last plan end_date = subscription_end_date (if present)
    grouped = subplans_df.groupby("subscription_id")
    for subscription_id, group in grouped:
        sub_end_raw = subs_end_map.get(subscription_id, None)
        if pd.isna(sub_end_raw) or sub_end_raw is None:
            # active subscription -> final plan keeps end_date = None (active)
            continue
        try:
            sub_end_date = pd.to_datetime(sub_end_raw).date()
        except Exception:
            # fallback: coerce
            sub_end_date = pd.to_datetime(sub_end_raw, errors="coerce").date()
        # find the last row index for this subscription
        last_idx = group.index.max()
        subplans_df.at[last_idx, "end_date"] = sub_end_date

    # Final touch: ensure types
    plans_df = plans_df.reset_index(drop=True)
    subplans_df = subplans_df.reset_index(drop=True)

    return plans_df, subplans_df
