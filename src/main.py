import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
# Load dataset
df = pd.read_csv("data/train.csv")

# Show basic info
print("Dataset Shape:", df.shape)

#print("\nFirst 5 Rows:")
#print(df.head())
#


#print("\nColumns:")
#print(df.columns)
#
#print("\nData Types:")
#print(df.dtypes)
#
#print("\nMissing Values:")
#print(df.isnull().sum().sort_values(ascending=False).head(10))



# -------------------------------
# Drop columns with too many missing values
# -------------------------------
missing_percent = df.isnull().mean() * 100

cols_to_drop = missing_percent[missing_percent > 80].index
df.drop(columns=cols_to_drop, inplace=True)

# -------------------------------
# Fill numerical columns
# -------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# -------------------------------
# Fill categorical columns
# -------------------------------
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("None")

# -------------------------------
# Check again
# -------------------------------
print("Missing values after cleaning:")
print(df.isnull().sum().sum())

# Drop ID column
df.drop(columns=["Id"], inplace=True)

# Convert categorical → numerical
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""# Train model
model = LinearRegression()
model.fit(X_train, y_train)
#
## Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse) """


#rf_model = RandomForestRegressor(random_state=42)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print("Random Forest RMSE:", rf_rmse)

joblib.dump(rf_model, "model/house_price_model.pkl")
print("Model saved successfully!")