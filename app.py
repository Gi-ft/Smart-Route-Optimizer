"""Streamlit dashboard for the Mwihoko six-customer route optimizer."""

from datetime import time
from html import escape
import math
from pathlib import Path
import sys

import folium
import networkx as nx
import pandas as pd
import streamlit as st
from folium.plugins import PolyLineTextPath
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from route_optimizer import optimize_routes
from scenario_analysis import (
    SCENARIOS,
    compare_scenarios,
    evaluate_route,
    optimize_by_scenario,
)
from graph_utils import normalize_edge_lengths
from route_intelligence import (
    assistant_answer_with_source,
    append_chat_message,
    build_context,
    evaluate_scenario,
    llm_configuration,
    parse_scenario_question,
)


st.set_page_config(
    page_title="Smart Route Optimizer",
    page_icon="🚚",
    layout="wide",
)

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]
DAY_INDEX = {day: index for index, day in enumerate(DAY_NAMES)}
VEHICLES = ["Motorcycle", "Car", "Van"]
VEHICLE_KEY = {vehicle: vehicle.lower() for vehicle in VEHICLES}


@st.cache_resource(show_spinner=False)
def load_road_network():
    """Load the expanded graph and location table once per app process."""
    graph_path = BASE_DIR / "data/raw/mwihoko_githurai_road_network_6_customers.graphml"
    locations_path = BASE_DIR / "data/raw/snapped_delivery_locations_6_customers.csv"
    # NetworkX reads the saved GraphML directly. OSMnx imports a large
    # geospatial stack that is only needed when regenerating the study graph.
    graph = nx.read_graphml(graph_path, node_type=int)
    graph.graph["invalid_edge_length_count"] = normalize_edge_lengths(graph)
    # GraphML may encode absent coordinates as the literal "-". Reject
    # malformed road geometry before a selected route reaches Folium.
    invalid_coordinate_nodes = []
    for node_id, attributes in graph.nodes(data=True):
        try:
            latitude = float(attributes["y"])
            longitude = float(attributes["x"])
            if not (math.isfinite(latitude) and math.isfinite(longitude)):
                raise ValueError
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError
            attributes["y"] = latitude
            attributes["x"] = longitude
        except (KeyError, TypeError, ValueError):
            invalid_coordinate_nodes.append(node_id)
    if invalid_coordinate_nodes:
        raise ValueError(
            "Road network contains missing or invalid coordinates for "
            f"{len(invalid_coordinate_nodes)} node(s); regenerate the saved graph."
        )

    locations = pd.read_csv(locations_path)
    if set(locations["location_id"]) != set("WABCDEF"):
        raise ValueError("The expanded location file must contain W and customers A–F.")
    for coordinate in ("latitude", "longitude"):
        locations[coordinate] = pd.to_numeric(locations[coordinate], errors="coerce")
    valid_locations = (
        locations["latitude"].map(math.isfinite)
        & locations["longitude"].map(math.isfinite)
        & locations["latitude"].between(-90, 90)
        & locations["longitude"].between(-180, 180)
    )
    if not valid_locations.all():
        bad_ids = locations.loc[~valid_locations, "location_id"].tolist()
        raise ValueError(
            "Delivery locations have missing or invalid coordinates for: "
            f"{', '.join(bad_ids)}. Correct the location data before mapping."
        )
    return graph, locations


@st.cache_data(show_spinner=False)
def cached_distance_routes():
    """Cache the 720 distance tours for repeated UI actions."""
    return optimize_routes()


@st.cache_data(show_spinner="Ranking 720 delivery sequences by estimated time…")
def cached_time_routes(vehicle, hour, day, scenario):
    return optimize_by_scenario(
        vehicle_type=vehicle,
        start_hour=hour,
        start_day=day,
        scenario=scenario,
    )


@st.cache_data(show_spinner=False)
def cached_scenario_comparison(vehicle, hour, day):
    return compare_scenarios(vehicle_type=vehicle, start_hour=hour, start_day=day)


def create_delivery_map(route=None):
    """Render locations and graph-derived road paths for the selected tour."""
    graph, locations = load_road_network()
    invalid_edges = graph.graph.get("invalid_edge_length_count", 0)
    if invalid_edges:
        st.warning(
            f"{invalid_edges} road-network edge(s) with invalid length data were "
            "skipped. Routes use the remaining valid roads."
        )
    node_by_id = {
        row.location_id: int(row.nearest_road_node)
        for row in locations.itertuples(index=False)
    }
    row_by_id = {row.location_id: row for row in locations.itertuples(index=False)}

    delivery_map = folium.Map(
        location=[locations["latitude"].mean(), locations["longitude"].mean()],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    bounds = []

    # Draw actual network paths first so markers remain easy to see.
    if route:
        route_color = "#0f766e"
        for leg_number, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
            road_nodes = nx.shortest_path(
                graph,
                source=node_by_id[origin],
                target=node_by_id[destination],
                weight="length",
            )
            coordinates = [
                [graph.nodes[node]["y"], graph.nodes[node]["x"]]
                for node in road_nodes
            ]
            bounds.extend(coordinates)
            line = folium.PolyLine(
                coordinates,
                color=route_color,
                weight=5,
                opacity=0.9,
                tooltip=f"Leg {leg_number}: {origin} → {destination}",
            ).add_to(delivery_map)
            PolyLineTextPath(
                line,
                "➤",
                repeat=True,
                offset=7,
                attributes={
                    "fill": route_color,
                    "font-weight": "bold",
                    "font-size": "16",
                },
            ).add_to(delivery_map)

    # Markers stay at the supplied place coordinates; a dashed connector shows
    # the offset to the graph node used for routing.
    ordered = route[:-1] if route else ["W", "A", "B", "C", "D", "E", "F"]
    order_number = {location_id: index for index, location_id in enumerate(ordered)}
    for location_id in ["W", "A", "B", "C", "D", "E", "F"]:
        row = row_by_id[location_id]
        site_point = [float(row.latitude), float(row.longitude)]
        node = graph.nodes[node_by_id[location_id]]
        road_point = [float(node["y"]), float(node["x"])]
        bounds.extend([site_point, road_point])

        # Snap distance is optional display metadata. An unavailable value
        # should suppress its connector/label, not be sent to Folium as a float.
        try:
            snap_distance = float(row.snap_distance_m)
            if not math.isfinite(snap_distance) or snap_distance < 0:
                snap_distance = None
        except (TypeError, ValueError):
            snap_distance = None
        if snap_distance is not None and snap_distance > 1:
            folium.PolyLine(
                [site_point, road_point],
                color="#94a3b8",
                weight=2,
                opacity=0.75,
                dash_array="5, 7",
                tooltip=f"Place coordinate to road node: {snap_distance:.1f} m",
            ).add_to(delivery_map)

        place_name = escape(str(row.location_name))
        sequence = order_number.get(location_id)
        marker_label = "W" if location_id == "W" else str(sequence or "•")
        marker_color = "#102a43" if location_id == "W" else "#0f766e"
        snap_description = (
            f"Road-node snap: {snap_distance:.1f} m"
            if snap_distance is not None
            else "Road-node snap distance unavailable"
        )
        popup_html = (
            f"<b>{place_name}</b><br>"
            f"{'Warehouse' if location_id == 'W' else f'Delivery stop {sequence}'}<br>"
            f"{snap_description}"
        )
        folium.Marker(
            site_point,
            tooltip=f"{location_id} · {row.location_name}",
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:{marker_color}; color:#fff; '
                    'border:2px solid white; border-radius:999px; width:30px; '
                    'height:30px; display:flex; align-items:center; '
                    'justify-content:center; font-weight:700; '
                    'box-shadow:0 2px 7px #0f172a66;">'
                    f"{marker_label}</div>"
                ),
                icon_size=(34, 34),
                icon_anchor=(17, 17),
            ),
        ).add_to(delivery_map)

    if bounds:
        delivery_map.fit_bounds(bounds, padding=(24, 24), max_zoom=15)
    return delivery_map


def display_route_sequence(route, names):
    """Format an ID route as a readable warehouse/customer sequence."""
    labels = []
    for index, location_id in enumerate(route):
        if location_id == "W":
            labels.append("Warehouse")
        else:
            labels.append(f"Stop {index} · {names[location_id]}")
    return " → ".join(labels)


def present_legs(leg_details, locations):
    names = dict(zip(locations["location_id"], locations["location_name"]))
    legs = pd.DataFrame(leg_details).copy()
    legs.insert(0, "Sequence", range(1, len(legs) + 1))
    legs["Origin"] = legs["origin"].map(names)
    legs["Destination"] = legs["destination"].map(names)
    legs = legs.rename(
        columns={
            "distance_km": "Road distance (km)",
            "predicted_eta_min": "Predicted ETA (min)",
            "departure_hour": "Departure hour",
        }
    )
    if "Departure hour" in legs:
        legs["Departure hour"] = legs["Departure hour"].map(
            lambda hour: f"{int(hour):02d}:00"
        )
    columns = [
        "Sequence", "Origin", "Destination", "Road distance (km)",
        "Predicted ETA (min)", "Departure hour",
    ]
    return legs[columns].round(
        {"Road distance (km)": 3, "Predicted ETA (min)": 2}
    )


# The compact native layout and branded theme keep the page responsive while
# using Streamlit's normal widgets, containers, and tables.
st.markdown(
    "<div style='font-size:.78rem; letter-spacing:.13em; text-transform:uppercase; "
    "color:#0f766e; font-weight:700;'>Mwihoko · Githurai · Delivery planning</div>",
    unsafe_allow_html=True,
)
st.title("Smart Route Optimizer")
st.subheader("Intelligent Delivery Routing and ETA Prediction System")
st.write(
    "Optimizing delivery sequences using road-network distances and "
    "machine-learning-based travel-time estimates."
)
st.divider()


with st.sidebar:
    st.header("Route configuration", anchor=False)
    with st.form("route_configuration", border=True):
        vehicle_label = st.selectbox("Vehicle type", VEHICLES, index=0)
        departure_day = st.selectbox("Day of the week", DAY_NAMES, index=0)
        departure_time = st.time_input("Departure time", value=time(17, 0), step=900)
        optimization_mode = st.segmented_control(
            "Optimization objective",
            ["Shortest Distance", "Fastest Delivery Time"],
            default="Shortest Distance",
            selection_mode="single",
        )
        st.caption("ETA sensitivity assumption")
        traffic_scenario = st.selectbox(
            "Traffic scenario",
            list(SCENARIOS),
            index=1,
            label_visibility="collapsed",
            help="Scenario sensitivity applied to simulated delay. This is not a live traffic feed.",
        )
        submitted = st.form_submit_button(
            "Optimize delivery route",
            type="primary",
            width="stretch",
            icon=":material/route:",
        )
    st.caption("One warehouse · Six delivery customers · Seven days supported")


graph, locations = load_road_network()
name_by_id = dict(zip(locations["location_id"], locations["location_name"]))

if submitted:
    vehicle = VEHICLE_KEY[vehicle_label]
    selected_hour = departure_time.hour
    selected_day = DAY_INDEX[departure_day]

    with st.spinner("Calculating the recommended delivery sequence…"):
        distance_routes = cached_distance_routes()
        distance_best = distance_routes.iloc[0]
        distance_evaluation = evaluate_route(
            route=distance_best["route"],
            vehicle_type=vehicle,
            start_hour=selected_hour,
            start_day=selected_day,
            scenario=traffic_scenario,
        )
        distance_eta = distance_evaluation[1]
        fastest_routes = cached_time_routes(
            vehicle, selected_hour, selected_day, traffic_scenario
        )
        fastest_result = fastest_routes[0]
        # Persist complete results for both objectives. The selector only
        # chooses the primary route to display; it does not skip either run.
        distance_result = {
            "route": list(distance_best["route"]),
            "route_string": distance_best["route_string"],
            "total_distance_km": float(distance_best["total_distance_km"]),
            "total_eta_min": float(distance_eta),
            "leg_details": distance_evaluation[2],
            "routes_evaluated": len(distance_routes),
        }
        time_result = {
            "route": list(fastest_result["route"]),
            "route_string": fastest_result["route_string"],
            "total_distance_km": float(fastest_result["total_distance_km"]),
            "total_eta_min": float(fastest_result["total_eta_min"]),
            "leg_details": fastest_result["leg_details"],
            "routes_evaluated": len(fastest_routes),
        }
        primary_key = (
            "time" if optimization_mode == "Fastest Delivery Time" else "distance"
        )
        primary_result = time_result if primary_key == "time" else distance_result
        st.session_state["optimization_result"] = {
            "primary_key": primary_key,
            "primary_result": primary_result,
            "distance_result": distance_result,
            "time_result": time_result,
            # Keep these aliases for existing display/session-state consumers.
            "route": primary_result["route"],
            "route_string": primary_result["route_string"],
            "total_distance": primary_result["total_distance_km"],
            "total_eta": primary_result["total_eta_min"],
            "vehicle_type": vehicle_label,
            "departure_day": departure_day,
            "departure_time": departure_time.strftime("%H:%M"),
            "optimization_mode": optimization_mode,
            "scenario": traffic_scenario,
            "leg_details": primary_result["leg_details"],
        }


result = st.session_state.get("optimization_result")
if result:
    st.caption(
        f"Recommended objective: {result['optimization_mode']} · "
        f"{result['vehicle_type']} · {result['departure_day']} "
        f"at {result['departure_time']} · {result['scenario']}"
    )
    metrics = st.columns(4, border=True)
    metrics[0].metric("Total route distance", f"{result['total_distance']:.2f} km")
    metrics[1].metric("Predicted delivery time", f"{result['total_eta']:.1f} min")
    metrics[2].metric("Customer stops", "6")
    metrics[3].metric("Selected vehicle", result["vehicle_type"])

    st.subheader("Recommended delivery sequence")
    st.markdown(
        " → ".join(
            f"**{label}**"
            for label in [
                "Warehouse · Mwihoko Shopping Centre"
                if location_id == "W"
                else f"Stop {index} · {name_by_id[location_id]}"
                for index, location_id in enumerate(result["route"])
            ]
        )
    )
else:
    st.info(
        "Choose the vehicle, departure time, day, and objective in the sidebar, "
        "then optimize to see route-specific estimates.",
        title="Ready to plan a delivery run",
    )


st.subheader("Delivery route map")
st.caption(
    "Markers show the supplied place coordinates; dashed connectors show the "
    "distance to their snapped road nodes. Route lines follow the driving graph."
)
try:
    delivery_map = create_delivery_map(result["route"] if result else None)
    st_folium(
        delivery_map,
        height=610,
        use_container_width=True,
        key="delivery_route_map",
        returned_objects=[],
    )
except Exception as error:
    st.error(f"Map could not be displayed: {error}", icon=":material/map:")


if result:
    st.subheader("Delivery leg breakdown")
    st.dataframe(
        present_legs(result["leg_details"], locations),
        hide_index=True,
        width="stretch",
        alt="Road distance and predicted ETA for all seven delivery legs, including the return to the warehouse.",
        column_config={
            "Sequence": st.column_config.NumberColumn(format="%d"),
            "Road distance (km)": st.column_config.NumberColumn(format="%.3f km"),
            "Predicted ETA (min)": st.column_config.NumberColumn(format="%.2f min"),
        },
    )

    st.subheader("Compare optimization objectives")
    st.caption(
        "Both routes use the same vehicle, departure time, day, warehouse, "
        "customers, and ETA sensitivity assumption."
    )
    if result["distance_result"]["route"] == result["time_result"]["route"]:
        st.info(
            "Both objectives selected the same delivery sequence for these settings; "
            "their distance and time scores are still evaluated independently."
        )
    comparison_columns = st.columns(2)
    objective_cards = [
        ("distance", "Shortest Distance", result["distance_result"]),
        ("time", "Fastest Delivery Time", result["time_result"]),
    ]
    for column, (objective_key, title, objective_result) in zip(
        comparison_columns, objective_cards
    ):
        with column.container(border=True):
            st.markdown(f"#### {title}")
            if objective_key == result["primary_key"]:
                st.success("Primary recommendation", icon=":material/check_circle:")
            else:
                st.caption("Alternative route")
            st.markdown(
                display_route_sequence(objective_result["route"], name_by_id)
            )
            distance_metric, eta_metric = st.columns(2)
            distance_metric.metric(
                "Total distance", f"{objective_result['total_distance_km']:.3f} km"
            )
            eta_metric.metric(
                "Predicted delivery time", f"{objective_result['total_eta_min']:.2f} min"
            )


st.divider()
assistant_context = build_context(
    result,
    locations,
    vehicle=result.get("vehicle_type") if result else vehicle_label,
    departure_day=result.get("departure_day") if result else departure_day,
    departure_time=result.get("departure_time") if result else departure_time.strftime("%H:%M"),
    objective=result.get("optimization_mode") if result else optimization_mode,
    scenario=result.get("scenario") if result else traffic_scenario,
)
st.session_state["route_assistant_context"] = assistant_context
llm_config = llm_configuration(st.secrets)
mode_options = ["Python analysis"] + (["LLM explanations (with Python fallback)"] if llm_config["available"] else [])
active_mode = st.session_state.get("assistant_response_mode", mode_options[0])
if active_mode not in mode_options:
    active_mode = mode_options[0]
    st.session_state["assistant_response_mode"] = active_mode

if "route_assistant_messages" not in st.session_state:
    st.session_state["route_assistant_messages"] = []
if "route_assistant_scenario_result" not in st.session_state:
    st.session_state["route_assistant_scenario_result"] = None
if "route_assistant_scenario_parameters" not in st.session_state:
    st.session_state["route_assistant_scenario_parameters"] = None
if "route_assistant_runtime" not in st.session_state:
    st.session_state["route_assistant_runtime"] = None
if "route_assistant_processing" not in st.session_state:
    st.session_state["route_assistant_processing"] = False

runtime = st.session_state.get("route_assistant_runtime") or {}
if not llm_config["available"] or active_mode == "Python analysis":
    status_label, status_kind = "Python analysis", "python"
elif runtime.get("requested_mode") == active_mode and runtime.get("source") == "openrouter":
    status_label, status_kind = "OpenRouter AI · Active", "ai"
elif runtime.get("requested_mode") == active_mode and runtime.get("source") == "scenario":
    status_label, status_kind = "Python scenario analysis", "python"
elif runtime.get("requested_mode") == active_mode and runtime.get("source") == "scope_guard":
    status_label, status_kind = "Route assistant", "python"
elif runtime.get("requested_mode") == active_mode and runtime.get("source") == "python":
    status_label, status_kind = "Python fallback · OpenRouter unavailable", "fallback"
else:
    status_label, status_kind = "OpenRouter · Ready", "ready"

with st.container(key="route_intelligence_copilot"):
    st.markdown(
        """
        <style>
        .st-key-route_intelligence_copilot {
            background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 150px);
            border: 1px solid #d9e4ec;
            border-radius: 22px;
            padding: 1.25rem 1.4rem 1.1rem;
            box-shadow: 0 12px 30px rgba(16, 42, 67, .07);
        }
        .st-key-route_intelligence_copilot .ric-heading {
            display:flex; align-items:center; gap:.85rem; color:#102a43;
            font-size:1.32rem; font-weight:750; letter-spacing:-.02em;
            line-height:1.2; margin:.15rem 0 .25rem;
        }
        .st-key-route_intelligence_copilot .ric-spark {
            display:grid; place-items:center; width:40px; height:40px; flex:none;
            border-radius:13px; background:#e2f6f6; color:#087f8c;
            font-size:1.35rem; border:1px solid #c5eceb;
        }
        .st-key-route_intelligence_copilot .ric-subtitle {
            margin-left:3.35rem; color:#587087; font-size:.9rem;
        }
        .st-key-route_intelligence_copilot .ric-status {
            display:inline-flex; align-items:center; gap:.45rem;
            padding:.38rem .7rem; border-radius:999px; font-size:.77rem;
            font-weight:650; border:1px solid #d5e2ea; color:#36536b;
            background:#fff; white-space:nowrap;
        }
        .st-key-route_intelligence_copilot .ric-status-dot {
            width:7px;height:7px;border-radius:50%;background:#0f766e;
        }
        .st-key-route_intelligence_copilot .ric-status-fallback .ric-status-dot {background:#b7791f;}
        .st-key-route_intelligence_copilot [data-testid="stChatMessage"] {
            border:1px solid #e0e8ee; background:#fff; border-radius:17px;
            padding:.85rem 1rem; margin:.6rem 0; box-shadow:0 3px 10px rgba(16,42,67,.035);
            max-width:94%;
        }
        .st-key-route_intelligence_copilot [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background:#e8f6f7; border-color:#cce9e9; margin-left:auto;
        }
        .st-key-route_intelligence_copilot [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            margin-right:auto;
        }
        .st-key-route_intelligence_copilot [data-testid="stChatMessage"] p {line-height:1.62;}
        .st-key-route_intelligence_copilot [data-testid="stChatMessage"] table {display:block;overflow-x:auto;}
        .st-key-route_intelligence_copilot .ric-welcome {
            padding:1.15rem 1.2rem; margin:.5rem 0 .9rem; background:#fff;
            border:1px solid #e0e8ee; border-radius:16px; color:#486176;
        }
        .st-key-route_intelligence_copilot .ric-section-label {
            color:#60788d; font-size:.75rem; font-weight:700;
            letter-spacing:.08em; text-transform:uppercase; margin:.85rem 0 .35rem;
        }
        .st-key-route_intelligence_copilot .ric-scenario {
            background:#f1f8fa; border:1px solid #cfe7eb; border-radius:17px;
            padding:1rem 1.1rem; margin:.8rem 0;
        }
        .st-key-route_intelligence_copilot .st-key-ric_scenario_results {
            background:#f1f8fa; border-color:#cfe7eb; border-radius:17px;
        }
        @media (max-width: 700px) {
            .st-key-route_intelligence_copilot {padding:.9rem;border-radius:16px;}
            .st-key-route_intelligence_copilot [data-testid="stChatMessage"] {max-width:100%;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    head, controls = st.columns([3.6, 2.4], vertical_alignment="center")
    with head:
        st.markdown(
            '<div class="ric-heading"><span class="ric-spark">✦</span>'
            '<span>ROUTE INTELLIGENCE COPILOT</span></div>'
            '<div class="ric-subtitle">Intelligent delivery route analysis and decision support</div>',
            unsafe_allow_html=True,
        )
    with controls:
        status_class = "ric-status-fallback" if status_kind == "fallback" else ""
        st.markdown(
            f'<div style="text-align:right"><span class="ric-status {status_class}">'
            f'<span class="ric-status-dot"></span>{status_label}</span></div>',
            unsafe_allow_html=True,
        )
        clear_col, mode_col = st.columns([1, 2.2], vertical_alignment="center")
        if clear_col.button("Clear chat", key="clear_route_assistant", width="stretch", help="Clear this conversation and scenario results"):
            st.session_state["route_assistant_messages"] = []
            st.session_state["route_assistant_scenario_result"] = None
            st.session_state["route_assistant_scenario_parameters"] = None
            st.session_state["route_assistant_runtime"] = None
            st.session_state["route_assistant_pending_prompt"] = None
            st.rerun()
        with mode_col:
            st.selectbox(
                "Response mode", mode_options, key="assistant_response_mode",
                help="OpenRouter runs only after you send a message. Python analysis remains available as fallback.",
                label_visibility="collapsed",
            )

    if assistant_context["ready"]:
        st.caption(
            f"Current context  ·  {assistant_context['vehicle']}  ·  "
            f"{assistant_context['departure_day']} {assistant_context['departure_time']}  ·  "
            f"{assistant_context['objective']}"
        )
        for warning in assistant_context["warnings"]:
            st.caption(f"Data note: {warning}")
    else:
        st.info(assistant_context["message"], icon=":material/route:")

    messages = st.session_state["route_assistant_messages"]
    if not messages:
        st.markdown(
            '<div class="ric-welcome"><strong>Plan with route intelligence</strong><br>'
            'Ask why a route was chosen, compare objectives, inspect delivery bottlenecks, '
            'or evaluate a what-if scenario. Answers use your current optimization results.</div>',
            unsafe_allow_html=True,
        )
    else:
        for message in messages:
            avatar = "🧑" if message["role"] == "user" else "✦"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    st.markdown('<div class="ric-section-label">Suggested questions</div>', unsafe_allow_html=True)
    suggested_questions = [
        "Why was this route selected?",
        "Compare shortest distance vs fastest time.",
        "Identify the slowest delivery leg.",
        "Explain the ETA prediction.",
        "What happens if I use a van?",
        "Recommend operational improvements.",
    ]
    prompt_columns = st.columns(3)
    for index, suggested in enumerate(suggested_questions):
        if prompt_columns[index % 3].button(
            suggested, key=f"assistant_suggestion_{index}", width="stretch",
            disabled=st.session_state["route_assistant_processing"],
        ):
            st.session_state["route_assistant_pending_prompt"] = suggested

    with st.form(
        "route_assistant_message_form",
        clear_on_submit=True,
        enter_to_submit=True,
        border=False,
    ):
        input_col, send_col = st.columns([6, 1.15], vertical_alignment="bottom")
        with input_col:
            typed_text = st.text_area(
                "Message the copilot", key="route_assistant_message_text",
                placeholder="Ask about this route or describe a what-if scenario…",
                height=94, label_visibility="collapsed",
            )
        with send_col:
            send_message = st.form_submit_button(
                "Thinking…" if st.session_state["route_assistant_processing"] else "Send",
                type="primary", icon=":material/arrow_upward:", width="stretch",
                disabled=st.session_state["route_assistant_processing"],
            )

    typed_question = typed_text.strip() if send_message and typed_text.strip() else None
    pending_prompt = st.session_state.get("route_assistant_pending_prompt")
    user_question = typed_question or pending_prompt
    if user_question:
        if typed_question and pending_prompt:
            st.session_state["route_assistant_pending_prompt"] = None
        elif pending_prompt:
            st.session_state.pop("route_assistant_pending_prompt", None)
        st.session_state["route_assistant_processing"] = True
        append_chat_message(st.session_state, "user", user_question)
        response = None
        response_source = "python"
        scenario_words = ("what if", "how would", "if i depart", "if i use", "prioritize distance", "prioritise distance", "prioritize time", "prioritise time", "instead of")
        if assistant_context["ready"] and any(term in user_question.lower() for term in scenario_words):
            response_source = "scenario"
            try:
                parameters = parse_scenario_question(user_question, {
                    **assistant_context,
                    "primary_key": result.get("primary_key", "distance"),
                })
                st.session_state["route_assistant_scenario_parameters"] = parameters
                with st.spinner("Copilot is calculating both route objectives…"):
                    scenario_result = evaluate_scenario(
                        parameters,
                        result,
                        optimize_distance=cached_distance_routes,
                        optimize_time=cached_time_routes,
                        evaluate_distance=evaluate_route,
                    )
                st.session_state["route_assistant_scenario_result"] = scenario_result
                active_key = "distance_result" if parameters["objective"] == "distance" else "time_result"
                scenario_primary = scenario_result[active_key]
                delta_distance = scenario_primary.get("delta_distance_km")
                delta_eta = scenario_primary.get("delta_eta_min")
                response = (
                    f"Scenario calculated for {parameters['vehicle']} departing "
                    f"{parameters['departure_day']} at {parameters['departure_hour']:02d}:00. "
                    f"The {parameters['objective']}-priority sequence is {scenario_primary['route_string']}: "
                    f"{scenario_primary['total_distance_km']:.2f} km and "
                    f"{scenario_primary['total_eta_min']:.2f} model-predicted min. "
                    + (f"Change from the current result: {delta_distance:+.2f} km and {delta_eta:+.2f} predicted min. "
                       if delta_distance is not None and delta_eta is not None else "A complete baseline delta is unavailable. ")
                    + "Both objectives are shown below. The dashboard changes only if you apply this scenario."
                )
            except ValueError as error:
                response = str(error)
            except Exception:
                response = "I couldn't calculate that scenario. Check the vehicle, weekday, and departure time, then try again. The current optimization remains unchanged."
        if response is None:
            response, response_source = assistant_answer_with_source(
                user_question,
                assistant_context,
                active_mode,
                llm_config,
                conversation_history=st.session_state["route_assistant_messages"][:-1],
            )
        st.session_state["route_assistant_runtime"] = {
            "requested_mode": active_mode,
            "source": response_source,
        }
        append_chat_message(st.session_state, "assistant", response)
        st.session_state["route_assistant_processing"] = False

    scenario_result = st.session_state.get("route_assistant_scenario_result")
    if scenario_result:
        parameters = scenario_result["parameters"]
        with st.container(key="ric_scenario_results", border=True):
            st.markdown("#### What-if scenario results")
            st.caption(
                f"{parameters['vehicle']} · {parameters['departure_day']} at "
                f"{parameters['departure_hour']:02d}:00 · "
                f"{'Shortest Distance' if parameters['objective'] == 'distance' else 'Fastest Delivery Time'} priority"
            )
            scenario_cards = st.columns(2)
            for col, (title, key) in zip(scenario_cards, (("Shortest Distance", "distance_result"), ("Fastest Delivery Time", "time_result"))):
                scenario_route = scenario_result[key]
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(f"Sequence: {scenario_route['route_string']}")
                    st.metric("Total distance", f"{scenario_route['total_distance_km']:.2f} km", delta=f"{scenario_route['delta_distance_km']:+.2f} km" if scenario_route["delta_distance_km"] is not None else None)
                    st.metric("Predicted duration", f"{scenario_route['total_eta_min']:.2f} min", delta=f"{scenario_route['delta_eta_min']:+.2f} min" if scenario_route["delta_eta_min"] is not None else None, delta_color="inverse")
            apply_col, _ = st.columns([1.6, 4])
            if apply_col.button("Apply scenario to dashboard", type="primary", key="apply_route_scenario", icon=":material/check:"):
                primary_key = parameters["objective"]
                primary_result = scenario_result[f"{primary_key}_result"]
                st.session_state["optimization_result"] = {
                    "primary_key": primary_key,
                    "primary_result": primary_result,
                    "distance_result": scenario_result["distance_result"],
                    "time_result": scenario_result["time_result"],
                    "route": primary_result["route"],
                    "route_string": primary_result["route_string"],
                    "total_distance": primary_result["total_distance_km"],
                    "total_eta": primary_result["total_eta_min"],
                    "vehicle_type": parameters["vehicle"],
                    "departure_day": parameters["departure_day"],
                    "departure_time": f"{parameters['departure_hour']:02d}:00",
                    "optimization_mode": "Shortest Distance" if primary_key == "distance" else "Fastest Delivery Time",
                    "scenario": parameters["scenario"],
                    "leg_details": primary_result["leg_details"],
                }
                st.rerun()


with st.expander("About the model and data", icon=":material/info:"):
    st.markdown(
        """
        - Road distances use paths from the OpenStreetMap driving network.
        - Dijkstra's algorithm ranks road distance for each origin–destination leg.
        - The ETA model predicts each leg using route and departure features; elapsed predicted travel time advances the departure hour for the next leg.
        - ETA training durations are **simulated**, not observed trips. Traffic sensitivity options are assumptions, not live traffic measurements.
        - Actual delivery times may differ because of traffic, service time, road conditions, access restrictions, or errors in mapped locations.
        """
    )
    naivas = locations.loc[locations["location_id"] == "D"].iloc[0]
    st.warning(
        f"Naivas Githurai 44 retains its supplied coordinates. Its nearest road node is "
        f"{naivas['snap_distance_m']:.1f} m away, so the location remains flagged for entrance verification.",
        title="Location accuracy note",
    )


with st.expander("Traffic sensitivity across delivery orders"):
    st.caption(
        "Ranks all 720 customer orders under three assumed delay factors. "
        "These are model sensitivity cases, not measured local traffic."
    )
    if st.button("Compare all 720 routes across traffic scenarios", icon=":material/compare_arrows:"):
        st.session_state["show_scenario_comparison"] = True
    if st.session_state.get("show_scenario_comparison", False):
        compare_vehicle = VEHICLE_KEY[vehicle_label]
        with st.spinner("Ranking scenario comparisons…"):
            scenario_df = cached_scenario_comparison(
                compare_vehicle, departure_time.hour, DAY_INDEX[departure_day]
            )
        st.dataframe(
            scenario_df,
            hide_index=True,
            width="stretch",
            alt="All route orders ranked across the traffic sensitivity scenarios.",
        )
