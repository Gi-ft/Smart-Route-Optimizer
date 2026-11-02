
"""
Smart Route Optimizer
Stage 10: Integrate route optimization and ETA prediction.
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from route_optimizer import optimize_routes
from eta_prediction import predict_eta


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DISTANCE_FILE = (
    BASE_DIR / "data" / "processed" /
    "distance_matrix_km.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "integrated_delivery_results.csv"
)


# --------------------------------------------------
# 2. Integration function
# --------------------------------------------------

def run_integrated_optimizer(
    vehicle_type="motorcycle",
    departure_hour=17,
    day_of_week=0
):

    # Validate starting time
    if not 0 <= departure_hour <= 23:
        raise ValueError("Departure hour must be between 0 and 23.")

    if not 0 <= day_of_week <= 6:
        raise ValueError("Day of week must be between 0 and 6.")

    # Obtain the optimal route
    routes = optimize_routes()

    if routes.empty:
        raise ValueError("No complete delivery route was found.")

    best_route = routes.iloc[0]["route"]

    # Load distance matrix
    distance_matrix = pd.read_csv(
        DISTANCE_FILE,
        index_col=0
    )

    # Create a reference Monday.
    # Monday = 0, Tuesday = 1, ..., Sunday = 6.
    current_time = (
        datetime(2026, 1, 5)
        + timedelta(days=day_of_week, hours=departure_hour)
    )

    leg_results = []
    total_eta = 0

    # --------------------------------------------------
    # 3. Predict ETA for every journey
    # --------------------------------------------------

    for i in range(len(best_route) - 1):

        origin = best_route[i]
        destination = best_route[i + 1]

        # Get journey distance
        distance = float(
            distance_matrix.loc[origin, destination]
        )

        # Use the current predicted departure time
        departure = current_time

        prediction = predict_eta(
            origin=origin,
            destination=destination,
            vehicle_type=vehicle_type,
            departure_hour=departure.hour,
            day_of_week=departure.weekday()
        )

        journey_eta = prediction[
            "predicted_delivery_duration_min"
        ]

        # Save journey details
        leg_results.append({
            "leg": i + 1,
            "origin": origin,
            "destination": destination,
            "departure_time": departure.strftime("%A %H:%M"),
            "distance_km": round(distance, 3),
            "predicted_eta_min": journey_eta,
            "arrival_time": (
                departure + timedelta(minutes=journey_eta)
            ).strftime("%A %H:%M")
        })

        # Update cumulative time
        total_eta += journey_eta

        current_time += timedelta(minutes=journey_eta)

    # --------------------------------------------------
    # 4. Prepare final results
    # --------------------------------------------------

    legs_df = pd.DataFrame(leg_results)

    total_distance = float(
        routes.iloc[0]["total_distance_km"]
    )

    # Save detailed results
    legs_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # 5. Display results
    # --------------------------------------------------

    print("\nSMART ROUTE OPTIMIZER")
    print("=" * 65)

    print("\nOPTIMAL DELIVERY ROUTE")
    print("-" * 65)

    print(" -> ".join(best_route))

    print("\nDELIVERY JOURNEY DETAILS")
    print("-" * 65)

    print(legs_df.to_string(index=False))

    print("\nFINAL DELIVERY SUMMARY")
    print("-" * 65)

    print(f"Vehicle type: {vehicle_type}")

    print(f"Total route distance: {total_distance:.3f} km")

    print(f"Total predicted journey time: {total_eta:.2f} minutes")

    print(
        "Estimated return time:",
        current_time.strftime("%A %H:%M")
    )

    print("\nDetailed results saved to:")
    print(OUTPUT_FILE)

    return legs_df


# --------------------------------------------------
# 6. Run the integrated system
# --------------------------------------------------

if __name__ == "__main__":

    run_integrated_optimizer(
        vehicle_type="motorcycle",
        departure_hour=17,
        day_of_week=0
    )
