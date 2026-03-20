import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create folders if not exist
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("data/INDIA_RETAIL_DATA_67.csv")

# Convert Order Date
data["Order Date"] = pd.to_datetime(data["Order Date"], errors='coerce')

# Group by State & City to calculate Sales Velocity
grouped = data.groupby(["State", "City"]).agg({
    "QtyOrdered": "mean",
    "Sales": "mean",
    "Profit": "mean"
}).reset_index()

grouped.rename(columns={"QtyOrdered": "Avg_Qty"}, inplace=True)

# Define Stock-out Risk (High demand = Risk)
median_demand = grouped["Avg_Qty"].median()

grouped["Stockout_Risk"] = np.where(
    grouped["Avg_Qty"] > median_demand, 1, 0
)

# Features
X = grouped[["Avg_Qty", "Sales", "Profit"]]
y = grouped["Stockout_Risk"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Save Model
joblib.dump(model, "model/stockout_model.pkl")
joblib.dump(median_demand, "model/median_value.pkl")

print("Model trained successfully!")