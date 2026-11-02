# Smart Route Optimizer

An interactive Streamlit dashboard for planning delivery routes around
Mwihoko and Githurai. The project compares shortest-distance and
time-aware routes using a trained ETA model and a local road network.

## Features

- Interactive delivery map with the selected road route
- Shortest-distance route optimization
- Fastest predicted-delivery-time optimization
- Vehicle-aware ETA estimates for motorcycles, cars, and vans
- Reproducible analysis scripts and generated evaluation data

## Setup

The project is tested with Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

The application uses the road network in `data/raw/`, processed distance
data in `data/processed/`, and the trained model in `models/`. Model files
are stored with Git LFS.

## Project layout

- `app.py` - Streamlit application entry point
- `src/` - data preparation, ETA prediction, and route optimization code
- `data/raw/` - source locations, road network, and simulated observations
- `data/processed/` - distance matrices, experiments, and evaluation output
- `models/` - trained ETA models
- `visualizations/` - generated analysis figures

## Notes

The `cache/` directory and local virtual environment are intentionally
ignored by Git. Recreate them locally as needed.
