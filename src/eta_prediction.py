
"""
ETA prediction module for the Smart Route Optimizer.

Loads the trained ETA model and predicts delivery duration
using route, vehicle, and departure-time information.
"""

from pathlib import Path
from functools import lru_cache

import joblib
import networkx as nx
import osmnx as ox
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "eta_model_6_customers.joblib"
DISTANCE_PATH = BASE_DIR / "data" / "processed" / "pairwise_distances_6_customers.csv"
LOCATION_PATH = BASE_DIR / "data" / "raw" / "snapped_delivery_locations_6_customers.csv"
GRAPH_PATH = BASE_DIR / "data" / "raw" / "mwihoko_githurai_road_network_6_customers.graphml"


# Assumed average speeds in km/h
VEHICLE_SPEEDS = {
    "motorcycle": 30,
    "car": 25,
    "van": 22
}


# Load resources
model = joblib.load(MODEL_PATH)

distances = pd.read_csv(DISTANCE_PATH)
locations = pd.read_csv(LOCATION_PATH)

G = ox.load_graphml(filepath=GRAPH_PATH)


@lru_cache(maxsize=50000)
def predict_eta(origin, destination, vehicle_type,
                departure_hour, day_of_week):

    # 1. Validate inputs
    if origin not in locations["location_id"].values:
        raise ValueError(f"Unknown origin: {origin}")

    if destination not in locations["location_id"].values:
        raise ValueError(f"Unknown destination: {destination}")

    if origin == destination:
        raise ValueError("Origin and destination must be different.")

    if vehicle_type not in VEHICLE_SPEEDS:
        raise ValueError("Vehicle must be motorcycle, car, or van.")

    if not 0 <= departure_hour <= 23:
        raise ValueError("Departure hour must be between 0 and 23.")

    if not 0 <= day_of_week <= 6:
        raise ValueError("Day of week must be between 0 and 6.")

    # 2. Retrieve shortest road distance
    distance_row = distances[
        (distances["origin"] == origin) &
        (distances["destination"] == destination)
    ]

    if distance_row.empty:
        raise ValueError("No distance found for this journey.")

    route_distance_km = float(
        distance_row.iloc[0]["distance_km"]
    )

    # 3. Retrieve road network nodes
    origin_node = int(
        locations.loc[
            locations["location_id"] == origin,
            "nearest_road_node"
        ].iloc[0]
    )

    destination_node = int(
        locations.loc[
            locations["location_id"] == destination,
            "nearest_road_node"
        ].iloc[0]
    )

    # 4. Calculate shortest-path edge count
    shortest_path = nx.shortest_path(
        G,
        source=origin_node,
        target=destination_node,
        weight="length"
    )

    road_edge_count = len(shortest_path) - 1

    # 5. Calculate estimated free-flow travel time
    speed = VEHICLE_SPEEDS[vehicle_type]

    estimated_free_flow_min = (
        route_distance_km / speed
    ) * 60

    # 6. Prepare model input
    input_data = pd.DataFrame([{
        "origin_zone": origin,
        "destination_zone": destination,
        "route_distance_km": route_distance_km,
        "road_edge_count": road_edge_count,
        "departure_hour": departure_hour,
        "day_of_week": day_of_week,
        "vehicle_type": vehicle_type,
        "estimated_free_flow_min": estimated_free_flow_min
    }])

    # 7. Predict delivery duration
    predicted_duration = model.predict(input_data)[0]

    return {
        "origin": origin,
        "destination": destination,
        "vehicle_type": vehicle_type,
        "route_distance_km": round(route_distance_km, 3),
        "road_edge_count": road_edge_count,
        "estimated_free_flow_min": round(
            estimated_free_flow_min, 2
        ),
        "predicted_delivery_duration_min": round(
            max(0, float(predicted_duration)), 2
        )
    }


# Example prediction
if __name__ == "__main__":

    result = predict_eta(
        origin="W",
        destination="A",
        vehicle_type="motorcycle",
        departure_hour=17,
        day_of_week=0
    )

    print("\nETA PREDICTION RESULT")
    print("---------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
