
"""
Time-Based Delivery Route Optimizer

Evaluates all possible delivery sequences and selects
the route with the lowest predicted journey time.
"""

from pathlib import Path
from itertools import permutations

import pandas as pd

from eta_prediction import predict_eta


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Delivery locations
DEPOT = "W"
CUSTOMERS = ["A", "B", "C"]


def calculate_route_eta(
    route,
    vehicle_type="motorcycle",
    start_hour=17,
    start_day=0
):

    current_hour = start_hour
    current_day = start_day

    total_eta = 0
    leg_details = []

    for i in range(len(route) - 1):

        origin = route[i]
        destination = route[i + 1]

        prediction = predict_eta(
            origin=origin,
            destination=destination,
            vehicle_type=vehicle_type,
            departure_hour=current_hour,
            day_of_week=current_day
        )

        eta = prediction["predicted_delivery_duration_min"]

        total_eta += eta

        leg_details.append({
            "origin": origin,
            "destination": destination,
            "departure_hour": current_hour,
            "day_of_week": current_day,
            "distance_km": prediction["route_distance_km"],
            "predicted_eta_min": eta
        })

        # Advance time after completing this leg
        total_minutes = current_hour * 60 + eta

        day_offset = int(total_minutes // 1440)

        current_day = (current_day + day_offset) % 7
        current_hour = int((total_minutes % 1440) // 60)

    total_distance = sum(
        leg["distance_km"] for leg in leg_details
    )

    return total_distance, total_eta, leg_details


def optimize_by_time(
    vehicle_type="motorcycle",
    start_hour=17,
    start_day=0
):

    results = []

    # Generate all six customer sequences
    for customer_order in permutations(CUSTOMERS):

        route = [DEPOT] + list(customer_order) + [DEPOT]

        distance, eta, details = calculate_route_eta(
            route=route,
            vehicle_type=vehicle_type,
            start_hour=start_hour,
            start_day=start_day
        )

        results.append({
            "route": route,
            "route_string": " -> ".join(route),
            "total_distance_km": distance,
            "total_eta_min": eta,
            "leg_details": details
        })

    # Rank routes by predicted total delivery time
    results.sort(key=lambda x: x["total_eta_min"])

    return results


if __name__ == "__main__":

    ranked_routes = optimize_by_time(
        vehicle_type="motorcycle",
        start_hour=17,
        start_day=0
    )

    print("\nTIME-BASED ROUTE OPTIMIZATION")
    print("=" * 70)

    for rank, result in enumerate(ranked_routes, start=1):

        print(f"\nRank: {rank}")
        print(f"Route: {result['route_string']}")
        print(f"Distance: {result['total_distance_km']:.3f} km")
        print(f"Predicted ETA: {result['total_eta_min']:.2f} minutes")

    fastest = ranked_routes[0]

    print("\nFASTEST DELIVERY ROUTE")
    print("=" * 70)
    print(f"Route: {fastest['route_string']}")
    print(f"Distance: {fastest['total_distance_km']:.3f} km")
    print(f"Predicted ETA: {fastest['total_eta_min']:.2f} minutes")

    # Save ranked results
    output = pd.DataFrame([
        {
            "Rank": rank,
            "Route": result["route_string"],
            "Total_Distance_km": round(result["total_distance_km"], 3),
            "Total_Predicted_ETA_min": round(result["total_eta_min"], 2)
        }
        for rank, result in enumerate(ranked_routes, start=1)
    ])

    output_path = (
        BASE_DIR / "data" / "processed" / "time_route_results.csv"
    )

    output.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
