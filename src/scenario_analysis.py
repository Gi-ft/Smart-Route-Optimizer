"""Compare delivery sequences under transparent traffic stress scenarios.

The scenarios scale the ETA model's predicted delay above free-flow time.
They are sensitivity assumptions, not measured traffic conditions.
"""

from itertools import permutations

import pandas as pd

from eta_prediction import predict_eta
from pathlib import Path


DEPOT = "W"
BASE_DIR = Path(__file__).resolve().parent.parent
LOCATIONS_FILE = BASE_DIR / "data" / "raw" / "snapped_delivery_locations_6_customers.csv"
CUSTOMERS = [
    location_id for location_id in pd.read_csv(LOCATIONS_FILE)["location_id"]
    if location_id != DEPOT
]
SCENARIOS = {
    "Light traffic": 0.5,
    "Typical simulated conditions": 1.0,
    "Heavy traffic": 1.5,
}


def evaluate_route(route, vehicle_type="motorcycle", start_hour=17,
                   start_day=0, scenario="Typical simulated conditions"):
    """Return distance, scenario-adjusted ETA, and leg-level details."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    current_minutes = start_hour * 60
    current_day = start_day
    total_distance = 0.0
    total_eta = 0.0
    details = []
    factor = SCENARIOS[scenario]

    for origin, destination in zip(route, route[1:]):
        hour = int((current_minutes % 1440) // 60)
        prediction = predict_eta(
            origin=origin,
            destination=destination,
            vehicle_type=vehicle_type,
            departure_hour=hour,
            day_of_week=current_day,
        )
        free_flow = prediction["estimated_free_flow_min"]
        predicted = prediction["predicted_delivery_duration_min"]
        excess_delay = max(0.0, predicted - free_flow)
        eta = round(free_flow + factor * excess_delay, 2)
        distance = prediction["route_distance_km"]

        details.append({
            "origin": origin,
            "destination": destination,
            "departure_hour": hour,
            "day_of_week": current_day,
            "distance_km": distance,
            "free_flow_min": round(free_flow, 2),
            "predicted_eta_min": eta,
            "scenario": scenario,
        })
        total_distance += distance
        total_eta += eta
        current_minutes += eta
        day_offset = int(current_minutes // 1440)
        current_day = (start_day + day_offset) % 7

    return total_distance, round(total_eta, 2), details


def optimize_by_scenario(vehicle_type="motorcycle", start_hour=17,
                         start_day=0, scenario="Typical simulated conditions"):
    """Rank all six stop sequences by their scenario-adjusted ETA."""
    results = []
    for order in permutations(CUSTOMERS):
        route = [DEPOT, *order, DEPOT]
        distance, eta, details = evaluate_route(
            route, vehicle_type, start_hour, start_day, scenario
        )
        results.append({
            "route": route,
            "route_string": " -> ".join(route),
            "total_distance_km": distance,
            "total_eta_min": eta,
            "leg_details": details,
            "scenario": scenario,
        })
    return sorted(results, key=lambda row: (row["total_eta_min"], row["total_distance_km"]))


def compare_scenarios(vehicle_type="motorcycle", start_hour=17, start_day=0):
    """Return rankings for all 720 routes in each traffic stress scenario."""
    rows = []
    for scenario in SCENARIOS:
        for rank, result in enumerate(
            optimize_by_scenario(vehicle_type, start_hour, start_day, scenario),
            start=1,
        ):
            rows.append({
                "scenario": scenario,
                "rank": rank,
                "route": result["route_string"],
                "total_distance_km": round(result["total_distance_km"], 3),
                "predicted_eta_min": result["total_eta_min"],
            })
    return pd.DataFrame(rows)
