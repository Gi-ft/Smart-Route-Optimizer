
"""
Smart Route Optimizer
Stage 7: Evaluate all possible delivery sequences.
"""

from pathlib import Path
from itertools import permutations

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DISTANCE_FILE = (
    BASE_DIR / "data" / "processed" / "distance_matrix_6_customers_km.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "route_optimization_results_6_customers.csv"
)


def optimize_routes():

    # Load distance matrix
    distance_matrix = pd.read_csv(
        DISTANCE_FILE,
        index_col=0
    )

    warehouse = "W"
    if warehouse not in distance_matrix.index:
        raise ValueError("Distance matrix does not contain the warehouse W.")
    customers = [location_id for location_id in distance_matrix.index if location_id != warehouse]
    if len(customers) != 6:
        raise ValueError(f"Expected six customers in the matrix; found {len(customers)}.")

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

    if len(results_df) != 720:
        raise RuntimeError(f"Expected 720 complete routes; found {len(results_df)}.")
    if not results_df["route"].map(
        lambda route: route[0] == warehouse and route[-1] == warehouse
        and len(route) == 8 and set(route[1:-1]) == set(customers)
    ).all():
        raise RuntimeError("A generated route does not visit all customers and return to W.")

    # Save results
    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return results_df


if __name__ == "__main__":

    results = optimize_routes()

    print("\nTOP FIVE DISTANCE-OPTIMAL DELIVERY ROUTES")
    print("-" * 55)

    print(
        results[
            ["rank", "route_string", "total_distance_km"]
        ].head(5).to_string(index=False)
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
