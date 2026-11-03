"""
Smart Route Optimizer
Stage 8A: Generate a simulated ETA training dataset.

IMPORTANT:
The journey durations are simulated, not real delivery observations.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

GRAPH_FILE = RAW_DIR / "mwihoko_githurai_road_network_6_customers.graphml"
LOCATIONS_FILE = RAW_DIR / "snapped_delivery_locations_6_customers.csv"

DISTANCE_FILE = PROCESSED_DIR / "distance_matrix_6_customers_km.csv"

OUTPUT_FILE = RAW_DIR / "simulated_eta_dataset_6_customers.csv"


# --------------------------------------------------
# 2. Simulation configuration
# --------------------------------------------------

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# Generate observations for every hour of the day.
DEPARTURE_HOURS = range(24)

DAYS_OF_WEEK = range(7)

REPETITIONS = 5


# Assumed average speeds in km/h.
VEHICLE_SPEEDS = {
    "motorcycle": 30,
    "car": 25,
    "van": 22
}


# Illustrative traffic multipliers for all 24 hours.
# Values are simulation assumptions, not measured traffic data.
TRAFFIC_MULTIPLIERS = {
    0: 0.85,
    1: 0.85,
    2: 0.85,
    3: 0.85,
    4: 0.85,
    5: 0.90,
    6: 1.25,
    7: 1.35,
    8: 1.30,
    9: 1.25,
    10: 1.15,
    11: 1.15,
    12: 1.05,
    13: 1.08,
    14: 1.08,
    15: 1.08,
    16: 1.35,
    17: 1.45,
    18: 1.40,
    19: 1.25,
    20: 1.10,
    21: 1.00,
    22: 0.90,
    23: 0.85
}

# Assumed day-of-week multipliers.
# Monday = 0, Sunday = 6.
DAY_MULTIPLIERS = {
    0: 1.00,
    1: 1.00,
    2: 1.00,
    3: 1.00,
    4: 1.05,
    5: 0.92,
    6: 0.90
}

# Assumed additional delay per road segment, in minutes.
EDGE_DELAY_MIN = 0.04


# --------------------------------------------------
# 3. Load road network and delivery locations
# --------------------------------------------------

print("Loading road network and delivery locations...")

G = ox.load_graphml(GRAPH_FILE)

locations = pd.read_csv(LOCATIONS_FILE)

distance_matrix = pd.read_csv(
    DISTANCE_FILE,
    index_col=0
)

locations["nearest_road_node"] = (
    locations["nearest_road_node"].astype(int)
)

location_nodes = dict(
    zip(
        locations["location_id"],
        locations["nearest_road_node"]
    )
)

print("Road nodes:", len(G.nodes))
print("Road edges:", len(G.edges))
print("Delivery locations:", len(location_nodes))


# --------------------------------------------------
# 4. Generate all directional journeys
# --------------------------------------------------

location_ids = list(location_nodes.keys())

journeys = []

for origin in location_ids:

    for destination in location_ids:

        if origin == destination:
            continue

        start_node = location_nodes[origin]
        end_node = location_nodes[destination]

        # Get the shortest road path.
        path = nx.shortest_path(
            G,
            source=start_node,
            target=end_node,
            weight="length",
            method="dijkstra"
        )

        # Get road distance from the existing matrix.
        distance_km = float(
            distance_matrix.loc[origin, destination]
        )

        # Count road segments in the shortest path.
        edge_count = len(path) - 1

        journeys.append({
            "origin_zone": origin,
            "destination_zone": destination,
            "route_distance_km": distance_km,
            "road_edge_count": edge_count
        })


print("\nDirectional journeys prepared:", len(journeys))


# --------------------------------------------------
# 5. Simulate delivery journey durations
# --------------------------------------------------

records = []

for journey in journeys:

    for day in DAYS_OF_WEEK:

        for hour in DEPARTURE_HOURS:

            for vehicle, speed in VEHICLE_SPEEDS.items():

                # Calculate free-flow travel time.
                free_flow_min = (
                    journey["route_distance_km"] / speed
                ) * 60

                # Apply traffic and day multipliers.
                traffic_multiplier = (
                    TRAFFIC_MULTIPLIERS[hour]
                    * DAY_MULTIPLIERS[day]
                )

                # Additional delay from road segments.
                edge_delay = (
                    journey["road_edge_count"]
                    * EDGE_DELAY_MIN
                )

                # Generate repeated observations.
                for repetition in range(REPETITIONS):

                    expected_duration = (
                        free_flow_min * traffic_multiplier
                        + edge_delay
                    )

                    # Add small random variation.
                    noise = rng.normal(
                        loc=0,
                        scale=max(0.5, expected_duration * 0.06)
                    )

                    simulated_duration = max(
                        1.0,
                        expected_duration + noise
                    )

                    records.append({
                        "origin_zone": journey["origin_zone"],
                        "destination_zone": journey["destination_zone"],
                        "route_distance_km": journey["route_distance_km"],
                        "road_edge_count": journey["road_edge_count"],
                        "departure_hour": hour,
                        "day_of_week": day,
                        "vehicle_type": vehicle,
                        "replicate_id": repetition + 1,
                        "estimated_free_flow_min": free_flow_min,
                        "delivery_duration_min": round(
                            simulated_duration, 8
                        )
                    })


# --------------------------------------------------
# 6. Create the dataset
# --------------------------------------------------

eta_data = pd.DataFrame(records)

expected_records = 42 * 7 * 24 * REPETITIONS * len(VEHICLE_SPEEDS)
expected_pairs = {
    (origin, destination)
    for origin in location_ids
    for destination in location_ids
    if origin != destination
}
actual_pairs = set(zip(eta_data["origin_zone"], eta_data["destination_zone"]))
if len(eta_data) != expected_records:
    raise RuntimeError(f"Expected {expected_records} records; got {len(eta_data)}.")
if actual_pairs != expected_pairs:
    raise RuntimeError("The ETA dataset does not contain all 42 directional pairs.")
if eta_data.isna().any().any():
    raise RuntimeError("The ETA dataset contains missing values.")
if eta_data.duplicated().any():
    raise RuntimeError("The ETA dataset contains duplicate records.")
if set(eta_data["vehicle_type"].unique()) != set(VEHICLE_SPEEDS):
    raise RuntimeError("The ETA dataset is missing a vehicle class.")
if set(eta_data["departure_hour"].unique()) != set(DEPARTURE_HOURS):
    raise RuntimeError("The ETA dataset is missing a departure hour.")

# Round numerical columns for readability.
eta_data["route_distance_km"] = (
    eta_data["route_distance_km"].round(3)
)

eta_data["estimated_free_flow_min"] = (
    eta_data["estimated_free_flow_min"].round(2)
)


# --------------------------------------------------
# 7. Validate the dataset
# --------------------------------------------------

print("\nDATASET VALIDATION")
print("-" * 45)

print("Number of records:", len(eta_data))
print("Number of columns:", len(eta_data.columns))

print("\nMissing values:")
print(eta_data.isnull().sum())

print("\nVehicle distribution:")
print(eta_data["vehicle_type"].value_counts())

print("\nDuration summary:")
print(eta_data["delivery_duration_min"].describe())


# --------------------------------------------------
# 8. Save the dataset
# --------------------------------------------------

eta_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset successfully saved to:")
print(OUTPUT_FILE)

print("\nFirst five records:")
print(eta_data.head())
