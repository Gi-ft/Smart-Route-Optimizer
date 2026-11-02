
"""
Stage 5: Road Network Extraction and Preparation

Project: Mwihoko Delivery Route Optimization
Source: OpenStreetMap
Libraries: OSMnx, NetworkX, Pandas
"""

from pathlib import Path
import math
import inspect

import pandas as pd
import osmnx as ox

# --------------------------------------------------
# 1. PROJECT DIRECTORIES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_FILE = RAW_DATA_DIR / "delivery_locations.csv"

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

    graph_file = (
        RAW_DATA_DIR / "mwihoko_githurai_road_network.graphml"
    )

    ox.save_graphml(G, filepath=graph_file)

    # Convert graph into GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    nodes.drop(columns="geometry").to_csv(
        RAW_DATA_DIR / "road_nodes.csv",
        index=True
    )

    edges.drop(columns="geometry").reset_index().to_csv(
        RAW_DATA_DIR / "road_edges.csv",
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

    output_file = (
        RAW_DATA_DIR / "snapped_delivery_locations.csv"
    )

    locations.to_csv(output_file, index=False)

    print("\nLOCATION MAPPING RESULTS")
    print("-" * 60)

    print(
        locations[
            [
                "location_id",
                "location_name",
                "nearest_road_node",
                "snap_distance_m"
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {output_file}")

    return locations


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    if not LOCATION_FILE.exists():
        raise FileNotFoundError(
            f"Location dataset not found: {LOCATION_FILE}"
        )

    locations = pd.read_csv(LOCATION_FILE)

    if len(locations) != 4:
        raise ValueError(
            "Expected one warehouse and three customers."
        )

    if locations["location_id"].nunique() != 4:
        raise ValueError("Location IDs must be unique.")

    print("DELIVERY LOCATIONS")
    print(locations.to_string(index=False))

    north, south, east, west = (
        calculate_bounding_box(locations)
    )

    G = download_road_network(
        north, south, east, west
    )

    save_road_network(G)

    snapped_locations = snap_locations_to_roads(
        G, locations
    )

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
