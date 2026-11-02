"""Application entry point for the smart route optimizer."""

import streamlit as st
import pandas as pd
from datetime import time
import folium
import networkx as nx
import osmnx as ox

from streamlit_folium import st_folium

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from time_route_optimizer import optimize_by_time
from route_optimizer import optimize_routes
# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="Smart Route Optimizer",
    page_icon="🚚",
    layout="wide"
)

# -----------------------------------
# DASHBOARD TITLE
# -----------------------------------
st.title("🚚 Smart Route Optimizer")
st.write(
    "An intelligent delivery planning system "
    "for route optimization and ETA prediction."
)

st.divider()

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.header("Delivery Configuration")

vehicle_type = st.sidebar.selectbox(
    "Select Vehicle",
    ["motorcycle", "car", "van"]
)

departure_day = st.sidebar.selectbox(
    "Departure Day",
    [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
)

departure_time = st.sidebar.time_input(
    "Departure Time",
    value=time(17, 0)
)

optimization_mode = st.sidebar.radio(
    "Optimization Objective",
    [
        "Shortest Distance",
        "Fastest Delivery Time"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Select your delivery preferences, "
    "then run the optimizer to generate a route."
)

# -----------------------------------
# DELIVERY LOCATIONS
# -----------------------------------
st.subheader("Delivery Network")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Warehouse", "Mwihoko")
col2.metric("Customer 1", "Mwihoko 46")
col3.metric("Customer 2", "Savory Supermarket")
col4.metric("Customer 3", "Stanmatt Githurai 45")

st.divider()

# -----------------------------------
# INTERACTIVE DELIVERY MAP
# -----------------------------------

@st.cache_resource
def load_road_network():

    graph_path = (
        BASE_DIR / "data" / "raw" /
        "mwihoko_githurai_road_network.graphml"
    )

    locations_path = (
        BASE_DIR / "data" / "raw" /
        "snapped_delivery_locations.csv"
    )

    graph = ox.load_graphml(filepath=graph_path)
    locations = pd.read_csv(locations_path)

    return graph, locations


def create_delivery_map(route):

    graph, locations = load_road_network()

    # Get the road network node for each location
    location_nodes = {
        row["location_id"]: int(row["nearest_road_node"])
        for _, row in locations.iterrows()
    }

    # Map centre
    map_center = [-1.207, 36.947]

    delivery_map = folium.Map(
        location=map_center,
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Location labels
    location_names = {
        "W": "Warehouse - Mwihoko Shopping Centre",
        "A": "Customer 1 - Mwihoko 46",
        "B": "Customer 2 - Savory Supermarket",
        "C": "Customer 3 - Stanmatt Githurai 45"
    }

    # Add warehouse and customer markers
    for location_id in ["W", "A", "B", "C"]:

        node = location_nodes[location_id]

        latitude = graph.nodes[node]["y"]
        longitude = graph.nodes[node]["x"]

        if location_id == "W":
            marker_color = "green"
        else:
            marker_color = "blue"

        folium.Marker(
            location=[latitude, longitude],
            popup=location_names[location_id],
            tooltip=f"{location_id}: {location_names[location_id]}",
            icon=folium.Icon(
                color=marker_color,
                icon="home" if location_id == "W" else "user"
            )
        ).add_to(delivery_map)

    # Draw the recommended road route
    route_colors = [
        "red", "blue", "purple", "orange"
    ]

    for i in range(len(route) - 1):

        origin = route[i]
        destination = route[i + 1]

        origin_node = location_nodes[origin]
        destination_node = location_nodes[destination]

        road_path = nx.shortest_path(
            graph,
            source=origin_node,
            target=destination_node,
            weight="length"
        )

        coordinates = [
            [
                graph.nodes[node]["y"],
                graph.nodes[node]["x"]
            ]
            for node in road_path
        ]

        folium.PolyLine(
            coordinates,
            color=route_colors[i % len(route_colors)],
            weight=5,
            opacity=0.8,
            tooltip=f"{origin} to {destination}"
        ).add_to(delivery_map)

    return delivery_map


# -----------------------------------
# RUN OPTIMIZER
# -----------------------------------

# -----------------------------------
# RUN OPTIMIZER
# -----------------------------------


# -----------------------------------
# RUN OPTIMIZER
# -----------------------------------

if st.button("🚀 Optimize Delivery Route", type="primary"):

    day_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    selected_day = day_mapping[departure_day]
    selected_hour = departure_time.hour

    with st.spinner("Optimizing delivery route..."):

        if optimization_mode == "Fastest Delivery Time":

            results = optimize_by_time(
                vehicle_type=vehicle_type,
                start_hour=selected_hour,
                start_day=selected_day
            )

            best_route = results[0]

            route = best_route["route"]
            route_string = best_route["route_string"]
            total_distance = best_route["total_distance_km"]
            total_eta = best_route["total_eta_min"]
            leg_details = best_route["leg_details"]

        else:

            results = optimize_routes()
            best_route = results.iloc[0]

            route = best_route["route"]
            route_string = best_route["route_string"]
            total_distance = best_route["total_distance_km"]

            from time_route_optimizer import calculate_route_eta

            _, total_eta, leg_details = calculate_route_eta(
                route=route,
                vehicle_type=vehicle_type,
                start_hour=selected_hour,
                start_day=selected_day
            )

        # Save results so they survive Streamlit reruns
        st.session_state["optimization_result"] = {
            "route": route,
            "route_string": route_string,
            "total_distance": total_distance,
            "total_eta": total_eta,
            "vehicle_type": vehicle_type,
            "leg_details": leg_details
        }


# -----------------------------------
# DISPLAY SAVED RESULTS
# -----------------------------------

if "optimization_result" in st.session_state:

    result = st.session_state["optimization_result"]

    st.success("Route optimization completed!")

    st.subheader("Recommended Delivery Route")
    st.write(f"**{result['route_string']}**")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Distance",
        f"{result['total_distance']:.3f} km"
    )

    col2.metric(
        "Predicted Delivery Time",
        f"{result['total_eta']:.2f} min"
    )

    col3.metric(
        "Selected Vehicle",
        result["vehicle_type"].title()
    )

    # -----------------------------------
    # JOURNEY BREAKDOWN
    # -----------------------------------

    st.subheader("Journey Breakdown")

    leg_df = pd.DataFrame(result["leg_details"])

    st.dataframe(
        leg_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------
    # INTERACTIVE DELIVERY MAP
    # -----------------------------------

    st.divider()
    st.subheader("🗺️ Recommended Delivery Route")

    try:

        delivery_map = create_delivery_map(result["route"])

        st_folium(
            delivery_map,
            height=550,
            use_container_width=True,
            key="delivery_route_map"
        )

    except Exception as e:

        st.error(f"Map could not be displayed: {e}")
