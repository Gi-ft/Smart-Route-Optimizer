
"""
Stage 4: Delivery Location Dataset
Mwihoko Delivery Route Optimization System

Coordinate source: Manually collected from Google Maps.
"""

from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Verified delivery locations
locations = [
    {
        "location_id": "W",
        "role": "Warehouse",
        "location_name": "Mwihoko Shopping Centre",
        "latitude": -1.2019324,
        "longitude": 36.9622615
    },
    {
        "location_id": "A",
        "role": "Customer",
        "location_name": "Mwihoko 46 Shopping Centre",
        "latitude": -1.2006509,
        "longitude": 36.9664823
    },
    {
        "location_id": "B",
        "role": "Customer",
        "location_name": "Savory Supermarket",
        "latitude": -1.2161198,
        "longitude": 36.9718889
    },
    {
        "location_id": "C",
        "role": "Customer",
        "location_name": "Stanmatt Supermarket Githurai 45",
        "latitude": -1.2056025,
        "longitude": 36.9187064
    },
    {
        "location_id": "D",
        "role": "Customer",
        "location_name": "Naivas Githurai 44",
        "latitude": -1.19875267519385,
        "longitude": 36.9005119783075
    },
    {
        "location_id": "E",
        "role": "Customer",
        "location_name": "Cleanshelf Kahawa Wendani",
        "latitude": -1.197502015958163,
        "longitude": 36.92697890999875
    },
    {
        "location_id": "F",
        "role": "Customer",
        "location_name": "Quickmart Kahawa Sukari",
        "latitude": -1.18919379482623,
        "longitude": 36.93237357931856
    }
]

def save_locations(output_name="delivery_locations_6_customers.csv"):

    df = pd.DataFrame(locations)

    expected_ids = set("WABCDEF")
    if set(df["location_id"]) != expected_ids:
        raise ValueError(
            f"Expected location IDs {sorted(expected_ids)}; "
            f"found {sorted(df['location_id'])}."
        )
    if (df["role"] == "Warehouse").sum() != 1:
        raise ValueError("The expanded network must contain one warehouse.")
    if (df["role"] == "Customer").sum() != 6:
        raise ValueError("The expanded network must contain six customers.")

    # Basic coordinate validation
    assert df["location_id"].is_unique
    assert df["latitude"].between(-90, 90).all()
    assert df["longitude"].between(-180, 180).all()
    assert df[["latitude", "longitude"]].notna().all().all()

    # Save dataset
    output_path = RAW_DATA_DIR / output_name

    df.to_csv(output_path, index=False)

    print("DELIVERY LOCATION DATASET")
    print("=" * 50)

    print(df.to_string(index=False))

    print("\nTotal locations:", len(df))
    print("Warehouse:", len(df[df["role"] == "Warehouse"]))
    print("Customers:", len(df[df["role"] == "Customer"]))

    print("\nDataset saved successfully!")
    print(output_path)


if __name__ == "__main__":
    save_locations()
