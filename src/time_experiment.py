
"""
Route and Departure-Time Experiment

Evaluates all six delivery sequences at four departure times.
Vehicle: Motorcycle
"""

from pathlib import Path
from itertools import permutations

import pandas as pd

from eta_prediction import predict_eta


BASE_DIR = Path(__file__).resolve().parent.parent

DEPOT = "W"
LOCATIONS_FILE = BASE_DIR / "data" / "raw" / "snapped_delivery_locations_6_customers.csv"
CUSTOMERS = [
    location_id for location_id in pd.read_csv(LOCATIONS_FILE)["location_id"]
    if location_id != DEPOT
]

VEHICLE = "motorcycle"
DEPARTURE_HOURS = [7, 12, 17, 20]
START_DAY = 0  # Monday


def evaluate_route(route, start_hour):

    current_hour = start_hour
    current_day = START_DAY
    elapsed_minutes = start_hour * 60

    total_eta = 0
    total_distance = 0

    for i in range(len(route) - 1):

        origin = route[i]
        destination = route[i + 1]

        prediction = predict_eta(
            origin=origin,
            destination=destination,
            vehicle_type=VEHICLE,
            departure_hour=current_hour,
            day_of_week=current_day
        )

        eta = prediction["predicted_delivery_duration_min"]
        distance = prediction["route_distance_km"]

        total_eta += eta
        total_distance += distance

        # Advance time after completing this journey leg
        elapsed_minutes += eta
        current_day = (START_DAY + int(elapsed_minutes // 1440)) % 7
        current_hour = int((elapsed_minutes % 1440) // 60)

    return total_distance, total_eta


def run_experiment():

    results = []

    # Evaluate every possible customer sequence
    for customer_order in permutations(CUSTOMERS):

        route = [DEPOT] + list(customer_order) + [DEPOT]

        for departure_hour in DEPARTURE_HOURS:

            distance, eta = evaluate_route(
                route,
                departure_hour
            )

            results.append({
                "Route": " -> ".join(route),
                "Departure_Hour": departure_hour,
                "Vehicle": VEHICLE,
                "Total_Distance_km": round(distance, 3),
                "Total_ETA_min": round(eta, 2)
            })

    results_df = pd.DataFrame(results)

    # Rank all 24 experiments
    results_df = results_df.sort_values(
        by="Total_ETA_min"
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Overall_Rank",
        range(1, len(results_df) + 1)
    )

    # Find the best route at each departure time
    best_by_time = (
        results_df.sort_values("Total_ETA_min")
        .groupby("Departure_Hour")
        .first()
        .reset_index()
        .sort_values("Departure_Hour")
    )

    # Save complete results
    output_path = (
        BASE_DIR / "data" / "processed"
        / "route_time_experiment_6_customers.csv"
    )

    results_df.to_csv(output_path, index=False)

    # Save best route for each departure time
    best_path = (
        BASE_DIR / "data" / "processed"
        / "best_route_by_departure_6_customers.csv"
    )

    best_by_time.to_csv(best_path, index=False)

    # Display results
    print("\nROUTE AND DEPARTURE-TIME EXPERIMENT")
    print("=" * 90)

    print("\nALL 24 EXPERIMENTS")
    print(results_df.to_string(index=False))

    print("\nBEST ROUTE AT EACH DEPARTURE TIME")
    print("=" * 90)
    print(best_by_time.to_string(index=False))

    print("\nOVERALL BEST DELIVERY STRATEGY")
    print("=" * 90)

    best = results_df.iloc[0]

    print(f"Route: {best['Route']}")
    print(f"Departure: {int(best['Departure_Hour']):02d}:00")
    print(f"Vehicle: {best['Vehicle']}")
    print(f"Distance: {best['Total_Distance_km']} km")
    print(f"Predicted ETA: {best['Total_ETA_min']} minutes")

    print(f"\nResults saved to: {output_path}")
    print(f"Departure summary saved to: {best_path}")


if __name__ == "__main__":
    run_experiment()
