"""Run end-to-end checks for the six-customer route optimizer."""

from pathlib import Path
import sys

import networkx as nx
import pandas as pd
import osmnx as ox

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from route_optimizer import optimize_routes
from time_route_optimizer import optimize_by_time
from eta_prediction import predict_eta

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOCATION_FILE = RAW_DIR / "snapped_delivery_locations_6_customers.csv"
GRAPH_FILE = RAW_DIR / "mwihoko_githurai_road_network_6_customers.graphml"
DISTANCE_FILE = PROCESSED_DIR / "distance_matrix_6_customers_km.csv"
DATASET_FILE = RAW_DIR / "simulated_eta_dataset_6_customers.csv"


def validate_data_and_routes():
    locations = pd.read_csv(LOCATION_FILE)
    graph = ox.load_graphml(GRAPH_FILE)
    matrix = pd.read_csv(DISTANCE_FILE, index_col=0)
    dataset = pd.read_csv(DATASET_FILE)

    assert len(locations) == 7 and locations["location_id"].nunique() == 7
    assert matrix.shape == (7, 7)
    assert not matrix.isna().any().any()
    pairs = pd.read_csv(PROCESSED_DIR / "pairwise_distances_6_customers.csv")
    assert len(pairs) == 42
    assert len(dataset) == 105840
    assert not dataset.isna().any().any()
    assert not dataset.duplicated().any()
    assert set(dataset["departure_hour"]) == set(range(24))
    assert set(dataset["vehicle_type"]) == {"motorcycle", "car", "van"}
    assert (
        set(dataset["origin_zone"]) | set(dataset["destination_zone"])
    ) == set("WABCDEF")
    assert not locations["nearest_road_node"].isna().any()

    node_map = dict(zip(
        locations["location_id"],
        locations["nearest_road_node"].astype(int),
    ))
    for origin, origin_node in node_map.items():
        for destination, destination_node in node_map.items():
            if origin != destination:
                assert nx.has_path(graph, origin_node, destination_node)

    distances = optimize_routes()
    assert len(distances) == 720
    customers = set("ABCDEF")
    assert distances["route"].map(
        lambda route: route[0] == "W" and route[-1] == "W"
        and len(route) == 8 and set(route[1:-1]) == customers
    ).all()

    # Exercise all 42 directional ETA lookups and validate the expanded IDs.
    for origin in "WABCDEF":
        for destination in "WABCDEF":
            if origin != destination:
                prediction = predict_eta(origin, destination, "motorcycle", 12, 0)
                assert prediction["predicted_delivery_duration_min"] > 0

    return distances


def validate_time_cases():
    cases = [
        ("motorcycle", 7, 0),
        ("motorcycle", 12, 0),
        ("motorcycle", 17, 0),
        ("car", 12, 0),
        ("van", 17, 0),
    ]
    rows = []
    for vehicle, hour, day in cases:
        ranked = optimize_by_time(vehicle_type=vehicle, start_hour=hour, start_day=day)
        assert len(ranked) == 720
        assert ranked[0]["route"][0] == "W" and ranked[0]["route"][-1] == "W"
        assert len(ranked[0]["leg_details"]) == 7
        rows.append({
            "vehicle": vehicle,
            "departure_hour": hour,
            "day_of_week": day,
            "fastest_route": ranked[0]["route_string"],
            "distance_km": round(ranked[0]["total_distance_km"], 3),
            "predicted_eta_min": round(ranked[0]["total_eta_min"], 2),
            "routes_evaluated": len(ranked),
        })
        print(
            f"{vehicle} {hour:02d}:00: {ranked[0]['route_string']} "
            f"({ranked[0]['total_eta_min']:.2f} min; 720 routes)"
        )
    output = PROCESSED_DIR / "time_validation_cases_6_customers.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    best_routes = validate_data_and_routes()
    print("Distance routes evaluated:", len(best_routes))
    print("Shortest route:", best_routes.iloc[0]["route_string"])
    print("Top five distance routes:")
    print(best_routes[["rank", "route_string", "total_distance_km"]].head(5).to_string(index=False))
    cases_file = validate_time_cases()
    print("Time-case results saved to:", cases_file)
    print("Data, graph connectivity, route enumeration, and ETA checks passed.")
