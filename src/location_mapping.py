
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
    }
]

def save_locations():

    df = pd.DataFrame(locations)

    # Basic coordinate validation
    assert df["location_id"].is_unique
    assert df["latitude"].between(-90, 90).all()
    assert df["longitude"].between(-180, 180).all()
    assert df[["latitude", "longitude"]].notna().all().all()

    # Save dataset
    output_path = RAW_DATA_DIR / "delivery_locations.csv"

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
