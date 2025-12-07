import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("dataset.csv")

X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = ["location", "floor_type"]
numeric_cols = ["num_rooms", "gross", "building_age", "furnishing_status"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor())
])

pipeline.fit(X, y)

joblib.dump(pipeline, "house_price_model.pkl")
print("Model trained successfully!")
