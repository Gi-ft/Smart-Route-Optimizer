
"""
Stage 6: Shortest Road Distance Matrix

Project: Mwihoko Delivery Route Optimization
Algorithm: Dijkstra's Shortest Path
"""

from pathlib import Path

import pandas as pd
import networkx as nx
import osmnx as ox

# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_FILE = (
    RAW_DATA_DIR /
    "mwihoko_githurai_road_network.graphml"
)

LOCATIONS_FILE = (
    RAW_DATA_DIR /
    "snapped_delivery_locations.csv"
)


# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

def load_data():

    print("Loading road network and locations...")

    G = ox.load_graphml(GRAPH_FILE)

    locations = pd.read_csv(LOCATIONS_FILE)

    locations["nearest_road_node"] = (
        locations["nearest_road_node"].astype(int)
    )

    print(f"Road nodes: {len(G.nodes)}")
    print(f"Road edges: {len(G.edges)}")
    print(f"Delivery locations: {len(locations)}")

    return G, locations


# -----------------------------------------
# 3. CALCULATE DISTANCES
# -----------------------------------------

def calculate_distance_matrix(G, locations):

    print("\nCalculating shortest road distances...")

    location_ids = locations["location_id"].tolist()

    node_mapping = dict(
        zip(
            locations["location_id"],
            locations["nearest_road_node"]
        )
    )

    matrix = pd.DataFrame(
        index=location_ids,
        columns=location_ids,
        dtype=float
    )

    detailed_results = []

    for origin in location_ids:

        for destination in location_ids:

            if origin == destination:

                matrix.loc[origin, destination] = 0
                continue

            start_node = node_mapping[origin]
            end_node = node_mapping[destination]

            try:

                distance = nx.shortest_path_length(
                    G,
                    source=start_node,
                    target=end_node,
                    weight="length",
                    method="dijkstra"
                )

                matrix.loc[origin, destination] = distance

                detailed_results.append({
                    "origin": origin,
                    "destination": destination,
                    "distance_m": distance,
                    "distance_km": distance / 1000
                })

                print(
                    f"{origin} -> {destination}: "
                    f"{distance / 1000:.3f} km"
                )

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound
            ):

                matrix.loc[origin, destination] = float("nan")

                print(
                    f"No available path: "
                    f"{origin} -> {destination}"
                )

    return matrix, detailed_results


# -----------------------------------------
# 4. SAVE RESULTS
# -----------------------------------------

def save_results(matrix, detailed_results):

    matrix_file = (
        PROCESSED_DIR / "distance_matrix_m.csv"
    )

    matrix_km_file = (
        PROCESSED_DIR / "distance_matrix_km.csv"
    )

    details_file = (
        PROCESSED_DIR / "pairwise_distances.csv"
    )

    matrix.to_csv(matrix_file)

    (matrix / 1000).to_csv(matrix_km_file)

    pd.DataFrame(detailed_results).to_csv(
        details_file,
        index=False
    )

    print("\nDISTANCE MATRIX (METRES)")
    print(matrix.round(2).to_string())

    print("\nDISTANCE MATRIX (KILOMETRES)")
    print((matrix / 1000).round(3).to_string())

    print("\nResults saved successfully!")


# -----------------------------------------
# 5. MAIN PROGRAMME
# -----------------------------------------

def main():

    G, locations = load_data()

    matrix, details = calculate_distance_matrix(
        G, locations
    )

    save_results(matrix, details)

    print("\nSTAGE 6 COMPLETED!")


if __name__ == "__main__":
    main()
