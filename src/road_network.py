
"""
Stage 5: Road Network Extraction and Preparation

Project: Mwihoko Delivery Route Optimization
Source: OpenStreetMap
Libraries: OSMnx, NetworkX, Pandas
"""

from pathlib import Path
import math
import inspect

import networkx as nx
import pandas as pd
import osmnx as ox

# --------------------------------------------------
# 1. PROJECT DIRECTORIES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_FILE = RAW_DATA_DIR / "delivery_locations_6_customers.csv"
GRAPH_FILE = RAW_DATA_DIR / "mwihoko_githurai_road_network_6_customers.graphml"
SNAPPED_FILE = RAW_DATA_DIR / "snapped_delivery_locations_6_customers.csv"
SNAP_WARNING_DISTANCE_M = 100

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

BUFFER = 0.01  # Approximately 1.1 km around the locations

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180

# --------------------------------------------------
# 3. CALCULATE BOUNDING BOX
# --------------------------------------------------

def calculate_bounding_box(df):

    north = df["latitude"].max() + BUFFER
    south = df["latitude"].min() - BUFFER

    east = df["longitude"].max() + BUFFER
    west = df["longitude"].min() - BUFFER

    print("\nSTUDY AREA BOUNDING BOX")
    print("-" * 40)
    print(f"North: {north}")
    print(f"South: {south}")
    print(f"East:  {east}")
    print(f"West:  {west}")

    return north, south, east, west


# --------------------------------------------------
# 4. DOWNLOAD ROAD NETWORK
# --------------------------------------------------

def download_road_network(north, south, east, west):

    print("\nDownloading road network from OpenStreetMap...")

    # Support both OSMnx 1.x and 2.x APIs
    parameters = inspect.signature(
        ox.graph_from_bbox
    ).parameters

    if "bbox" in parameters:

        # OSMnx 2.x
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type="drive",
            simplify=True
        )

    else:

        # OSMnx 1.x
        G = ox.graph_from_bbox(
            north,
            south,
            east,
            west,
            network_type="drive",
            simplify=True
        )

    if len(G.nodes) == 0:
        raise RuntimeError("Downloaded road network is empty.")

    print("\nRoad network downloaded!")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")

    return G


# --------------------------------------------------
# 5. SAVE ROAD NETWORK
# --------------------------------------------------

def save_road_network(G):

    print("\nSaving road network files...")

    ox.save_graphml(G, filepath=GRAPH_FILE)

    # Convert graph into GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    nodes.drop(columns="geometry").to_csv(
        RAW_DATA_DIR / "road_nodes_6_customers.csv",
        index=True
    )

    edges.drop(columns="geometry").reset_index().to_csv(
        RAW_DATA_DIR / "road_edges_6_customers.csv",
        index=False
    )

    print("GraphML and CSV files saved successfully.")


# --------------------------------------------------
# 6. CALCULATE DISTANCE BETWEEN COORDINATES
# --------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371000

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * math.asin(math.sqrt(a))


# --------------------------------------------------
# 7. MAP DELIVERY LOCATIONS TO ROAD NODES
# --------------------------------------------------

def snap_locations_to_roads(G, locations):

    print("\nMapping delivery locations to road network...")

    nearest_nodes = ox.distance.nearest_nodes(
        G,
        X=locations["longitude"].to_numpy(),
        Y=locations["latitude"].to_numpy()
    )

    locations = locations.copy()

    locations["nearest_road_node"] = nearest_nodes

    snap_distances = []

    for _, row in locations.iterrows():

        node = G.nodes[row["nearest_road_node"]]

        distance = haversine_distance(
            row["latitude"],
            row["longitude"],
            node["y"],
            node["x"]
        )

        snap_distances.append(distance)

    locations["snap_distance_m"] = snap_distances

    locations["snap_warning"] = locations["snap_distance_m"] > SNAP_WARNING_DISTANCE_M
    locations.to_csv(SNAPPED_FILE, index=False)

    print("\nLOCATION MAPPING RESULTS")
    print("-" * 60)

    print(
        locations[
            [
                "location_id",
                "location_name",
                "nearest_road_node",
                "snap_distance_m",
                "snap_warning"
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {SNAPPED_FILE}")

    return locations


def validate_location_connectivity(G, snapped_locations):
    """Confirm every ordered pair is connected in the directed road graph."""
    node_by_id = dict(zip(
        snapped_locations["location_id"],
        snapped_locations["nearest_road_node"].astype(int),
    ))
    failures = []
    for origin, origin_node in node_by_id.items():
        for destination, destination_node in node_by_id.items():
            if origin == destination:
                continue
            try:
                nx.shortest_path_length(
                    G, origin_node, destination_node, weight="length"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                failures.append(f"{origin}->{destination}")
    if failures:
        raise RuntimeError(
            "Disconnected delivery pairs in road graph: " + ", ".join(failures)
        )
    print("All 42 directional delivery paths are connected.")


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    if not LOCATION_FILE.exists():
        raise FileNotFoundError(
            f"Location dataset not found: {LOCATION_FILE}"
        )

    locations = pd.read_csv(LOCATION_FILE)

    if len(locations) != 7:
        raise ValueError(
            "Expected one warehouse and six customers."
        )

    if locations["location_id"].nunique() != 7:
        raise ValueError("Location IDs must be unique.")

    print("DELIVERY LOCATIONS")
    print(locations.to_string(index=False))

    north, south, east, west = (
        calculate_bounding_box(locations)
    )

    # Always extract a fresh graph for the expanded extent. The legacy graph
    # remains untouched and is not used as the six-customer network.
    G = download_road_network(
        north, south, east, west
    )

    save_road_network(G)

    snapped_locations = snap_locations_to_roads(
        G, locations
    )
    validate_location_connectivity(G, snapped_locations)

    print("\n" + "=" * 50)
    print("STAGE 5 COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"Total road nodes: {len(G.nodes)}")
    print(f"Total road edges: {len(G.edges)}")
    print(
        f"Delivery locations mapped: "
        f"{len(snapped_locations)}"
    )


if __name__ == "__main__":
    main()
