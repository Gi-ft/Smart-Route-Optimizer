"""Inspect and map the Naivas Githurai 44 road-network snap.

This is a diagnostic only: it preserves the supplied delivery coordinate and
does not rewrite the snapped locations, graph, or distance matrix.
"""

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
from folium.plugins import PolyLineTextPath
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import nearest_points


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "visualizations"
GRAPH_FILE = RAW_DIR / "mwihoko_githurai_road_network_6_customers.graphml"
LOCATIONS_FILE = RAW_DIR / "delivery_locations_6_customers.csv"
SNAPPED_FILE = RAW_DIR / "snapped_delivery_locations_6_customers.csv"
MAP_FILE = OUTPUT_DIR / "naivas_snap_diagnostic.html"
REPORT_FILE = OUTPUT_DIR / "naivas_snap_diagnostic.csv"


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_371_000 * 2 * asin(sqrt(a))


def main():
    graph = ox.load_graphml(filepath=GRAPH_FILE)
    locations = pd.read_csv(LOCATIONS_FILE).set_index("location_id")
    snapped = pd.read_csv(SNAPPED_FILE).set_index("location_id")
    store = locations.loc["D"]
    snapped_row = snapped.loc["D"]

    nearest_node = ox.distance.nearest_nodes(
        graph, X=float(store.longitude), Y=float(store.latitude)
    )
    node_data = graph.nodes[nearest_node]
    node_distance = haversine_m(
        store.latitude, store.longitude, node_data["y"], node_data["x"]
    )

    # Project locally to UTM 37S so point-to-road geometry measurements use metres.
    projected_crs = "EPSG:32737"
    point_wgs84 = Point(float(store.longitude), float(store.latitude))
    point_projected = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(
        projected_crs
    ).iloc[0]
    _, edges = ox.graph_to_gdfs(graph)
    edges_projected = edges.to_crs(projected_crs)
    edges_projected["gap_m"] = edges_projected.geometry.distance(point_projected)
    nearest_edge = edges_projected.sort_values("gap_m").iloc[0]
    edge_geometry_projected = nearest_edge.geometry
    edge_point_projected = nearest_points(point_projected, edge_geometry_projected)[1]
    to_wgs84 = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    edge_lon, edge_lat = to_wgs84.transform(edge_point_projected.x, edge_point_projected.y)

    nearest_edge_wgs84 = gpd.GeoSeries(
        [nearest_edge.geometry], crs=edges_projected.crs
    ).to_crs("EPSG:4326").iloc[0]
    nearest_edge_coords = [
        [lat, lon] for lon, lat in nearest_edge_wgs84.coords
    ]

    # Preserve directions while suppressing duplicate parallel-edge drawing.
    unique = set()
    nearby = []
    buffer = point_projected.buffer(450)
    for (u, v, key), row in edges_projected.iterrows():
        if not row.geometry.intersects(buffer):
            continue
        edge_key = (min(int(u), int(v)), max(int(u), int(v)), str(row.get("name")))
        if edge_key in unique:
            continue
        unique.add(edge_key)
        geometry = gpd.GeoSeries(
            [row.geometry], crs=edges_projected.crs
        ).to_crs("EPSG:4326").iloc[0]
        coords = [[lat, lon] for lon, lat in geometry.coords]
        nearby.append((coords, row))

    delivery_map = folium.Map(
        location=[float(store.latitude), float(store.longitude)],
        zoom_start=17,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    for coords, edge in nearby:
        name = edge.get("name")
        highway = edge.get("highway")
        label = f"{name or 'Unnamed road'} · {highway or 'road'}"
        folium.PolyLine(
            coords,
            color="#94a3b8",
            weight=3,
            opacity=0.75,
            tooltip=label,
        ).add_to(delivery_map)

    folium.PolyLine(
        [[float(store.latitude), float(store.longitude)], [float(node_data["y"]), float(node_data["x"])]],
        color="#dc2626",
        weight=4,
        dash_array="8, 6",
        tooltip=f"Supplied point to nearest graph node: {node_distance:.1f} m",
    ).add_to(delivery_map)
    folium.PolyLine(
        [[float(store.latitude), float(store.longitude)], [edge_lat, edge_lon]],
        color="#f59e0b",
        weight=4,
        dash_array="3, 6",
        tooltip=f"Supplied point to nearest mapped road segment: {nearest_edge.gap_m:.1f} m",
    ).add_to(delivery_map)
    nearest_edge_line = folium.PolyLine(
        nearest_edge_coords,
        color="#0f766e",
        weight=6,
        opacity=0.95,
        tooltip=f"Nearest segment: {nearest_edge.get('name')} ({nearest_edge.get('highway')})",
    ).add_to(delivery_map)
    PolyLineTextPath(
        nearest_edge_line,
        "➤",
        repeat=True,
        offset=6,
        attributes={"fill": "#0f766e", "font-weight": "bold"},
    ).add_to(delivery_map)

    folium.Marker(
        [float(store.latitude), float(store.longitude)],
        tooltip="Supplied Naivas coordinate (retained)",
        popup=(
            "Naivas Githurai 44 supplied point<br>"
            f"{store.latitude:.8f}, {store.longitude:.8f}"
        ),
        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
    ).add_to(delivery_map)
    folium.Marker(
        [float(node_data["y"]), float(node_data["x"])],
        tooltip=f"Nearest routable graph node {nearest_node}",
        popup=(
            f"Node {nearest_node}<br>"
            f"{node_distance:.1f} m from supplied coordinate<br>"
            "Connected road: residential (OSM graph tag)"
        ),
        icon=folium.Icon(color="blue", icon="road", prefix="fa"),
    ).add_to(delivery_map)
    folium.Marker(
        [edge_lat, edge_lon],
        tooltip=f"Nearest road line: {nearest_edge.get('name')} · {nearest_edge.get('highway')}",
        popup=(
            f"Nearest mapped road segment<br>Name: {nearest_edge.get('name')}<br>"
            f"OSM highway class: {nearest_edge.get('highway')}<br>"
            f"Line distance: {nearest_edge.gap_m:.1f} m"
        ),
        icon=folium.Icon(color="green", icon="road", prefix="fa"),
    ).add_to(delivery_map)
    delivery_map.fit_bounds(
        [[float(store.latitude), float(store.longitude)], [float(node_data["y"]), float(node_data["x"])], [edge_lat, edge_lon]],
        padding=(30, 30),
    )

    road_names = nearest_edge.get("name")
    road_classes = nearest_edge.get("highway")
    nearest_node_edges = [
        {
            "road_name": data.get("name"),
            "highway": data.get("highway"),
            "access": data.get("access"),
            "service": data.get("service"),
            "length_m": data.get("length"),
        }
        for _, _, _, data in graph.edges(nearest_node, keys=True, data=True)
    ]
    report = pd.DataFrame(
        [
            {
                "location_id": "D",
                "location_name": store.location_name,
                "supplied_latitude": store.latitude,
                "supplied_longitude": store.longitude,
                "nearest_node_id": nearest_node,
                "nearest_node_latitude": node_data["y"],
                "nearest_node_longitude": node_data["x"],
                "node_snap_distance_m": node_distance,
                "nearest_node_road_edges": nearest_node_edges,
                "nearest_road_name": road_names,
                "nearest_road_highway_class": road_classes,
                "nearest_road_segment_gap_m": float(nearest_edge.gap_m),
                "nearest_road_access_tag": nearest_edge.get("access"),
                "existing_snap_file_distance_m": float(snapped_row.snap_distance_m),
                "coordinate_changed": False,
                "entrance_verified": False,
            }
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_FILE, index=False)
    delivery_map.save(MAP_FILE)

    print(report.drop(columns="nearest_node_road_edges").T.to_string(header=False))
    print(f"Nearest node edge tags: {nearest_node_edges}")
    print(f"Diagnostic map saved to {MAP_FILE}")
    print(f"Diagnostic report saved to {REPORT_FILE}")
    print("No supplied coordinates or routing datasets were changed.")


if __name__ == "__main__":
    main()
