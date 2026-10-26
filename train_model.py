
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "starter_kit" / "train_telemetry.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load training data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Remove invalid rows
df = df.dropna(
    subset=[
        "timestamp",
        "ambient_temp_c",
        "humidity_pct",
        "solar_irradiance_wm2",
        "target_demand_kw",
    ]
).copy()

# Extract useful time features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

features = [
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
]
target = "target_demand_kw"

X = df[features]
y = df[target]

# Chronological split: later observations are held out
split_index = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Histogram Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=42,
    ),
}

results = []
best_model = None
best_name = None
best_rmse = float("inf")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    })

    print(f"RMSE: {rmse:.3f} kW")
    print(f"MAE : {mae:.3f} kW")
    print(f"R2  : {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name

# Save model comparison
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

# Save the best model and its feature names
joblib.dump(best_model, MODEL_DIR / "demand_model.joblib")
joblib.dump(features, MODEL_DIR / "feature_names.joblib")

print("\n" + "=" * 45)
print("TRAINING COMPLETE")
print("=" * 45)
print(f"Best model: {best_name}")
print(f"Best holdout RMSE: {best_rmse:.3f} kW")
print(f"Saved model: {MODEL_DIR / 'demand_model.joblib'}")
print(f"Saved comparison: {MODEL_DIR / 'model_comparison.csv'}")

