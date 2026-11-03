"""Generate the default traffic sensitivity table for the study."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from scenario_analysis import compare_scenarios


def main():
    results = compare_scenarios(
        vehicle_type="motorcycle",
        start_hour=17,
        start_day=0,
    )
    output = BASE_DIR / "data" / "processed" / "scenario_route_comparison_6_customers.csv"
    results.to_csv(output, index=False)
    print(
        results.groupby("scenario", sort=False).head(5).to_string(index=False)
    )
    print(f"\nScenario comparison saved to: {output}")


if __name__ == "__main__":
    main()
