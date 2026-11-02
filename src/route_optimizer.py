
"""
Smart Route Optimizer
Stage 7: Evaluate all possible delivery sequences.
"""

from pathlib import Path
from itertools import permutations

import pandas as pd

from time_route_optimizer import optimize_by_time


BASE_DIR = Path(__file__).resolve().parent.parent

DISTANCE_FILE = (
    BASE_DIR / "data" / "processed" / "distance_matrix_km.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "route_optimization_results.csv"
)


def optimize_routes():

    # Load distance matrix
    distance_matrix = pd.read_csv(
        DISTANCE_FILE,
        index_col=0
    )

    warehouse = "W"
    customers = ["A", "B", "C"]

    # Generate all possible delivery sequences
    delivery_sequences = list(permutations(customers))

    results = []

    # Calculate total distance for each route
    for sequence in delivery_sequences:

        route = [warehouse] + list(sequence) + [warehouse]

        total_distance = 0

        for i in range(len(route) - 1):

            origin = route[i]
            destination = route[i + 1]

            distance = distance_matrix.loc[
                origin, destination
            ]

            if pd.isna(distance):
                total_distance = None
                break

            total_distance += distance

        results.append({
            "route": route,
            "route_string": " -> ".join(route),
            "total_distance_km": total_distance
        })

    # Rank routes
    results_df = pd.DataFrame(results)

    results_df = results_df.dropna(
        subset=["total_distance_km"]
    )

    results_df = results_df.sort_values(
        by="total_distance_km"
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1)
    )

    # Save results
    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return results_df


if __name__ == "__main__":

    results = optimize_routes()

    print("\nDELIVERY ROUTE OPTIMIZATION RESULTS")
    print("-" * 55)

    print(
        results[
            ["rank", "route_string", "total_distance_km"]
        ].to_string(index=False)
    )

    if not results.empty:

        best_route = results.iloc[0]

        print("\nOPTIMAL DELIVERY ROUTE")
        print("-" * 55)

        print("Route:", best_route["route_string"])

        print(
            f"Total distance: "
            f"{best_route['total_distance_km']:.3f} km"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Test another configuration
    print("\nCAR AT NOON")

    results = optimize_by_time(
        vehicle_type="car",
        start_hour=12,
        start_day=0
    )

    print(results[0]["route_string"])
    print(results[0]["total_eta_min"])
