"""Prepare collected data for modeling."""
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "simulated_eta_dataset_6_customers.csv"
)

df = pd.read_csv(DATA_FILE)

# --------------------------------------------------
# 2. Basic dataset information
# --------------------------------------------------

print("\nDATASET OVERVIEW")
print("=" * 50)

print("Shape:", df.shape)
print("Number of records:", len(df))
print("Number of features:", len(df.columns) - 1)

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\nMISSING VALUE ANALYSIS")
print("=" * 50)

print(df.isnull().sum())

# --------------------------------------------------
# 4. Duplicate records
# --------------------------------------------------

print("\nDUPLICATE ANALYSIS")
print("=" * 50)

print("Duplicate records:", df.duplicated().sum())

# --------------------------------------------------
# 5. Check numerical ranges
# --------------------------------------------------

print("\nNUMERICAL SUMMARY")
print("=" * 50)

print(df.describe().round(2))

# --------------------------------------------------
# 6. Check categorical values
# --------------------------------------------------

print("\nCATEGORICAL VALUES")
print("=" * 50)

for column in [
    "origin_zone",
    "destination_zone",
    "vehicle_type",
    "day_of_week",
    "departure_hour"
]:
    print(f"\n{column}:")
    print(sorted(df[column].unique()))

# --------------------------------------------------
# 7. Check invalid values
# --------------------------------------------------

print("\nVALIDATION CHECKS")
print("=" * 50)

print(
    "Non-positive durations:",
    (df["delivery_duration_min"] <= 0).sum()
)

print(
    "Non-positive distances:",
    (df["route_distance_km"] <= 0).sum()
)

print(
    "Invalid departure hours:",
    (~df["departure_hour"].between(0, 23)).sum()
)

print(
    "Invalid road edge counts:",
    (df["road_edge_count"] <= 0).sum()
)

# --------------------------------------------------
# 8. Average duration by vehicle
# --------------------------------------------------

print("\nAVERAGE DURATION BY VEHICLE")
print("=" * 50)

print(
    df.groupby("vehicle_type")[
        "delivery_duration_min"
    ].mean().round(2)
)

# --------------------------------------------------
# 9. Average duration by distance
# --------------------------------------------------

print("\nAVERAGE DURATION BY JOURNEY")
print("=" * 50)

print(
    df.groupby(
        ["origin_zone", "destination_zone"]
    )["delivery_duration_min"]
    .mean()
    .round(2)
)

print("\nDataset validation completed.")
