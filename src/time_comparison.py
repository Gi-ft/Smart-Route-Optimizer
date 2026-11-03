
"""
Compare predicted delivery times at different departure hours.
Vehicle: Motorcycle
Route: W -> A -> B -> C -> W
"""

from pathlib import Path
import pandas as pd

from eta_prediction import predict_eta
from route_optimizer import optimize_routes

BASE_DIR = Path(__file__).resolve().parent.parent

# Departure times to compare
DEPARTURE_HOURS = [7, 12, 17, 20]

# Motorcycle fixed for a fair comparison
VEHICLE = "motorcycle"

# Monday = 0
START_DAY = 0


def compare_departure_times():

    routes = optimize_routes()
    best_route = routes.iloc[0]["route"]
    total_distance = float(routes.iloc[0]["total_distance_km"])

    results = []

    for start_hour in DEPARTURE_HOURS:

        current_hour = start_hour
        current_day = START_DAY
        elapsed_minutes = start_hour * 60
        total_eta = 0

        for i in range(len(best_route) - 1):

            origin = best_route[i]
            destination = best_route[i + 1]

            prediction = predict_eta(
                origin=origin,
                destination=destination,
                vehicle_type=VEHICLE,
                departure_hour=current_hour,
                day_of_week=current_day
            )

            eta = prediction["predicted_delivery_duration_min"]

            total_eta += eta

            # Advance absolute elapsed time after each journey leg.
            elapsed_minutes += eta
            current_day = (START_DAY + int(elapsed_minutes // 1440)) % 7
            current_hour = int((elapsed_minutes % 1440) // 60)

        results.append({
            "Departure_Hour": start_hour,
            "Vehicle": VEHICLE,
            "Total_Distance_km": round(total_distance, 3),
            "Total_Predicted_ETA_min": round(total_eta, 2)
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Total_Predicted_ETA_min"
    ).reset_index(drop=True)

    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    output_path = (
        BASE_DIR / "data" / "processed" / "time_comparison_6_customers.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nDEPARTURE TIME COMPARISON")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nFastest departure time:")
    print(f"{int(results_df.iloc[0]['Departure_Hour']):02d}:00")

    print(
        "\nLowest predicted journey time:",
        results_df.iloc[0]["Total_Predicted_ETA_min"],
        "minutes"
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_departure_times()
