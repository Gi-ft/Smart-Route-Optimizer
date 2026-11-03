"""
Compare ETA predictions for different vehicle types
on the optimized delivery route.
"""

from pathlib import Path
import pandas as pd

from eta_prediction import predict_eta
from route_optimizer import optimize_routes


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Vehicle types
VEHICLES = ["motorcycle", "car", "van"]

# Starting conditions
START_HOUR = 17
START_DAY = 0  # Monday


def compare_vehicles():

    # Get the optimal route
    routes = optimize_routes()
    best_route = routes.iloc[0]["route"]
    total_distance = float(routes.iloc[0]["total_distance_km"])

    results = []

    for vehicle in VEHICLES:

        current_hour = START_HOUR
        current_day = START_DAY
        elapsed_minutes = START_HOUR * 60
        total_eta = 0

        for i in range(len(best_route) - 1):

            origin = best_route[i]
            destination = best_route[i + 1]

            # Predict journey duration
            prediction = predict_eta(
                origin=origin,
                destination=destination,
                vehicle_type=vehicle,
                departure_hour=current_hour,
                day_of_week=current_day
            )
            eta = prediction["predicted_delivery_duration_min"]

            total_eta += eta

            # Advance departure time
            elapsed_minutes += eta
            current_day = (START_DAY + int(elapsed_minutes // 1440)) % 7
            current_hour = int((elapsed_minutes % 1440) // 60)

        results.append({
            "Vehicle": vehicle,
            "Total_Distance_km": round(total_distance, 3),
            "Total_Predicted_ETA_min": round(total_eta, 2)
        })

    results_df = pd.DataFrame(results)

    # Rank vehicles by predicted journey time
    results_df = results_df.sort_values(
        by="Total_Predicted_ETA_min"
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Rank",
        range(1, len(results_df) + 1)
    )

    # Save results
    output_path = (
        BASE_DIR / "data" / "processed" / "vehicle_comparison_6_customers.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nVEHICLE COMPARISON RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nFastest predicted vehicle:")
    print(results_df.iloc[0]["Vehicle"])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_vehicles()
