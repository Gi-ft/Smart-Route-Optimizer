
"""
Smart Route Optimizer
Stage 8B: Train, evaluate, compare and save ETA models.

Target:
    delivery_duration_min

Models:
    1. Median Baseline
    2. Linear Regression
    3. Random Forest
    4. Gradient Boosting

IMPORTANT:
The dataset contains simulated delivery durations.
Performance metrics therefore measure performance on simulated data,
not proven real-world ETA accuracy.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw" / "simulated_eta_dataset_6_customers.csv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

print("DATASET INFORMATION")
print("-" * 55)
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(
    "Departure hours:",
    sorted(df["departure_hour"].unique())
)


# --------------------------------------------------
# 2. Create scenario groups
# --------------------------------------------------

# The five repeated observations belonging to one
# scenario must remain in the same data partition.

df["scenario_group"] = (
    df["origin_zone"].astype(str) + "_" +
    df["destination_zone"].astype(str) + "_" +
    df["day_of_week"].astype(str) + "_" +
    df["departure_hour"].astype(str) + "_" +
    df["vehicle_type"].astype(str)
)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

target = "delivery_duration_min"

categorical_features = [
    "origin_zone",
    "destination_zone",
    "departure_hour",
    "day_of_week",
    "vehicle_type"
]

numerical_features = [
    "route_distance_km",
    "road_edge_count",
    "estimated_free_flow_min"
]

features = categorical_features + numerical_features

X = df[features]
y = df[target]

groups = df["scenario_group"]


# --------------------------------------------------
# 4. Define preprocessing
# --------------------------------------------------

def create_preprocessor(numeric_columns):

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer(transformers=[
        ("num", numerical_transformer, numeric_columns),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ])


# --------------------------------------------------
# 5. Define model factory
# --------------------------------------------------

def create_models():

    return {
        "Median Baseline": DummyRegressor(
            strategy="median"
        ),

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=20,
            max_depth=14,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=25,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE
        )
    }


# --------------------------------------------------
# 6. Group-based train-test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nSCENARIO-BASED TRAIN-TEST SPLIT")
print("-" * 55)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training scenarios:",
    groups.iloc[train_idx].nunique()
)

print(
    "Testing scenarios:",
    groups.iloc[test_idx].nunique()
)


# --------------------------------------------------
# 7. Train and compare four models
# --------------------------------------------------

model_results = []

for model_name, model in create_models().items():

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor(
            numerical_features
        )),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    model_results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{model_name}")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 4))


model_results_df = pd.DataFrame(model_results)

model_results_df = model_results_df.sort_values(
    by="MAE"
)

print("\nSCENARIO-BASED MODEL COMPARISON")
print(model_results_df.to_string(index=False))

model_results_df.to_csv(
    PROCESSED_DIR / "eta_model_comparison_6_customers.csv",
    index=False
)


# --------------------------------------------------
# 8. Feature ablation experiment
# --------------------------------------------------

# Investigate the contribution of estimated free-flow time.

feature_experiments = {
    "With Free Flow": numerical_features,

    "Without Free Flow": [
        "route_distance_km",
        "road_edge_count"
    ]
}

experiment_results = []

for experiment_name, numeric_columns in (
    feature_experiments.items()
):

    experiment_features = (
        categorical_features + numeric_columns
    )

    X_train_exp = X_train[experiment_features]
    X_test_exp = X_test[experiment_features]

    for model_name, model in create_models().items():

        if model_name == "Median Baseline":
            continue

        pipeline = Pipeline(steps=[
            (
                "preprocessor",
                create_preprocessor(numeric_columns)
            ),
            ("model", model)
        ])

        pipeline.fit(X_train_exp, y_train)

        predictions = pipeline.predict(X_test_exp)

        experiment_results.append({
            "Feature_Set": experiment_name,
            "Model": model_name,
            "MAE": mean_absolute_error(
                y_test, predictions
            ),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "R2": r2_score(y_test, predictions)
        })


experiment_df = pd.DataFrame(experiment_results)

print("\nFEATURE ABLATION RESULTS")
print(
    experiment_df.sort_values(
        by=["Feature_Set", "MAE"]
    ).to_string(index=False)
)

experiment_df.to_csv(
    PROCESSED_DIR / "eta_feature_comparison_6_customers.csv",
    index=False
)


# --------------------------------------------------
# 9. Unseen-route holdout experiment
# --------------------------------------------------

# Group by directional route.
# All observations for a route stay together.

route_groups = (
    df["origin_zone"].astype(str)
    + "_TO_"
    + df["destination_zone"].astype(str)
)

route_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

route_train_idx, route_test_idx = next(
    route_splitter.split(
        X,
        y,
        groups=route_groups
    )
)

X_route_train = X.iloc[route_train_idx]
X_route_test = X.iloc[route_test_idx]

y_route_train = y.iloc[route_train_idx]
y_route_test = y.iloc[route_test_idx]

training_routes = sorted(
    route_groups.iloc[route_train_idx].unique()
)

testing_routes = sorted(
    route_groups.iloc[route_test_idx].unique()
)

print("\nUNSEEN-ROUTE HOLDOUT EXPERIMENT")
print("-" * 55)

print("Training routes:", training_routes)
print("Testing routes:", testing_routes)

print("Training records:", len(X_route_train))
print("Testing records:", len(X_route_test))


route_holdout_results = []

for model_name, model in create_models().items():

    pipeline = Pipeline(steps=[
        (
            "preprocessor",
            create_preprocessor(numerical_features)
        ),
        ("model", model)
    ])

    pipeline.fit(X_route_train, y_route_train)

    predictions = pipeline.predict(X_route_test)

    route_holdout_results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(
            y_route_test, predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_route_test, predictions
            )
        ),
        "R2": r2_score(
            y_route_test, predictions
        )
    })


route_holdout_df = pd.DataFrame(route_holdout_results)

route_holdout_df = route_holdout_df.sort_values(
    by="MAE"
)

print("\nUNSEEN-ROUTE HOLDOUT RESULTS")
print(route_holdout_df.to_string(index=False))

route_holdout_df.to_csv(
    PROCESSED_DIR / "eta_unseen_route_comparison_6_customers.csv",
    index=False
)


# --------------------------------------------------
# 10. Four-fold route-level cross-validation
# --------------------------------------------------

# Each fold tests on complete directional routes
# that were excluded from its training data.

route_cv = GroupKFold(n_splits=4)

cv_results = []

for model_name, model in create_models().items():

    if model_name == "Median Baseline":
        continue

    fold_maes = []

    actual_values = []
    predicted_values = []

    for fold, (train_idx, test_idx) in enumerate(
        route_cv.split(
            X,
            y,
            groups=route_groups
        ),
        start=1
    ):

        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]

        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        pipeline = Pipeline(steps=[
            (
                "preprocessor",
                create_preprocessor(numerical_features)
            ),
            ("model", clone(model))
        ])

        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict(X_test_fold)

        fold_mae = mean_absolute_error(
            y_test_fold,
            predictions
        )

        fold_maes.append(fold_mae)

        actual_values.extend(
            y_test_fold.tolist()
        )

        predicted_values.extend(
            predictions.tolist()
        )

        print(
            f"{model_name} - Fold {fold}: "
            f"MAE = {fold_mae:.3f}"
        )

    # Calculate pooled out-of-fold metrics.
    overall_mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    overall_r2 = r2_score(
        actual_values,
        predicted_values
    )

    cv_results.append({
        "Model": model_name,
        "Mean_Fold_MAE": np.mean(fold_maes),
        "Std_Fold_MAE": np.std(fold_maes),
        "Overall_MAE": overall_mae,
        "Overall_RMSE": overall_rmse,
        "Overall_R2": overall_r2
    })


cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    by="Overall_MAE"
)

print("\nFOUR-FOLD UNSEEN-ROUTE VALIDATION")
print("-" * 55)

print(cv_results_df.to_string(index=False))

cv_results_df.to_csv(
    PROCESSED_DIR / "eta_route_cross_validation_6_customers.csv",
    index=False
)


# --------------------------------------------------
# 11. Select and save the final ETA model
# --------------------------------------------------

# Balance generalization evidence from the unseen-route holdout and
# route-grouped cross-validation. R-squared is reported but not used to
# select the model. Mean ranks limit the effect of different split sizes.
holdout_ranks = route_holdout_df.set_index("Model")["MAE"].rank(method="average")
cv_ranks = cv_results_df.set_index("Model")["Mean_Fold_MAE"].rank(method="average")
selection_df = pd.DataFrame({
    "Model": cv_results_df["Model"],
    "Unseen_Route_Holdout_MAE": cv_results_df["Model"].map(
        route_holdout_df.set_index("Model")["MAE"]
    ),
    "Route_CV_Mean_Fold_MAE": cv_results_df["Model"].map(
        cv_results_df.set_index("Model")["Mean_Fold_MAE"]
    ),
    "Unseen_Route_Holdout_Rank": cv_results_df["Model"].map(holdout_ranks),
    "Route_CV_Rank": cv_results_df["Model"].map(cv_ranks),
})
selection_df["Mean_Generalization_Rank"] = selection_df[
    ["Unseen_Route_Holdout_Rank", "Route_CV_Rank"]
].mean(axis=1)
selection_df = selection_df.sort_values(
    ["Mean_Generalization_Rank", "Unseen_Route_Holdout_MAE", "Route_CV_Mean_Fold_MAE"]
).reset_index(drop=True)
selection_df.to_csv(
    PROCESSED_DIR / "eta_model_selection_6_customers.csv", index=False
)

best_model_name = selection_df.iloc[0]["Model"]
selected_cv = cv_results_df.loc[
    cv_results_df["Model"] == best_model_name
].iloc[0]

print("\nFINAL MODEL SELECTION")
print("-" * 55)

print("Selected model:", best_model_name)

selected_model = create_models()[best_model_name]

final_pipeline = Pipeline(steps=[
    (
        "preprocessor",
        create_preprocessor(numerical_features)
    ),
    ("model", selected_model)
])

# Fit the selected model using the complete dataset.
final_pipeline.fit(X, y)

model_path = MODEL_DIR / "eta_model_6_customers.joblib"

joblib.dump(final_pipeline, model_path)

# Save the model selection information.
selection_summary = {
    "selected_model": best_model_name,
    "selection_metric": "Mean rank across unseen-route holdout MAE and route-grouped CV fold MAE",
    "unseen_route_holdout_mae": float(
        selection_df.iloc[0]["Unseen_Route_Holdout_MAE"]
    ),
    "route_cv_mean_fold_mae": float(
        selection_df.iloc[0]["Route_CV_Mean_Fold_MAE"]
    ),
    "cross_validation_mae": float(
        selected_cv["Overall_MAE"]
    ),
    "cross_validation_rmse": float(
        selected_cv["Overall_RMSE"]
    ),
    "cross_validation_r2": float(
        selected_cv["Overall_R2"]
    ),
    "training_records": len(X),
    "dataset_type": "Simulated"
}

pd.DataFrame([selection_summary]).to_csv(
    PROCESSED_DIR / "eta_final_model_selection_6_customers.csv",
    index=False
)

print("\nFINAL ETA MODEL")
print("Algorithm:", best_model_name)
print("Training records:", len(X))
print("Model saved to:", model_path)

print("\nAll model evaluation results saved successfully.")
