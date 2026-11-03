from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "raw" / "simulated_eta_dataset_6_customers.csv"
)

OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)


# --------------------------------------------------
# 2. Summary statistics
# --------------------------------------------------

print("\nETA EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nAverage duration by vehicle:")
print(
    df.groupby("vehicle_type")[
        "delivery_duration_min"
    ].mean().round(2)
)

print("\nAverage duration by departure hour:")
print(
    df.groupby("departure_hour")[
        "delivery_duration_min"
    ].mean().round(2)
)

print("\nCorrelation matrix:")
print(
    df.select_dtypes(include="number")
    .corr()
    .round(2)
)


# --------------------------------------------------
# 3. Distribution of delivery duration
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["delivery_duration_min"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Simulated Delivery Durations")
plt.xlabel("Delivery Duration (minutes)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "eta_duration_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Distance versus delivery duration
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    df["route_distance_km"],
    df["delivery_duration_min"],
    alpha=0.4
)

plt.title("Route Distance vs Delivery Duration")
plt.xlabel("Route Distance (km)")
plt.ylabel("Delivery Duration (minutes)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distance_vs_duration.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 5. Delivery duration by vehicle
# --------------------------------------------------

plt.figure(figsize=(9, 5))

vehicle_order = ["motorcycle", "car", "van"]

vehicle_data = [
    df.loc[
        df["vehicle_type"] == vehicle,
        "delivery_duration_min"
    ].values
    for vehicle in vehicle_order
]

plt.boxplot(
    vehicle_data,
    tick_labels=vehicle_order
)

plt.title("Delivery Duration by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Duration (minutes)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "duration_by_vehicle.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 6. Departure hour versus duration
# --------------------------------------------------

hourly_duration = (
    df.groupby("departure_hour")[
        "delivery_duration_min"
    ].mean()
)

plt.figure(figsize=(9, 5))

plt.plot(
    hourly_duration.index,
    hourly_duration.values,
    marker="o"
)

plt.title("Average Delivery Duration by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Average Duration (minutes)")
plt.xticks(hourly_duration.index)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "duration_by_departure_hour.png",
    dpi=300
)

plt.show()


print("\nEDA completed.")
print("Visualizations saved in:", OUTPUT_DIR)
