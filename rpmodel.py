import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 1. Load the dataset
file_path = 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx'
df = pd.read_excel(file_path, sheet_name='Training_Data')

# 2. Prepare features and target using the ENTIRE dataset (Valid + Invalid)
features = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 
            'Test_Duration_min', 'Sensor_S1', 'Sensor_S2', 'Sensor_S3']
target = 'Reference_Parameter'

X = df[features].copy()
y = df[target]

# 3. Impute missing sensor values with column medians 
# (Crucial for handling missing/corrupted data in invalid rows before regression)
for col in ['Sensor_S1', 'Sensor_S2', 'Sensor_S3']:
    X[col] = X[col].fillna(X[col].median())

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Gradient Boosting Regressor on all data
gbr_full = GradientBoostingRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
gbr_full.fit(X_train, y_train)

# 5. Evaluate predictions across all data types
preds = gbr_full.predict(X_val)

print("--- Full Dataset Regression Performance ---")
print(f"R² Score: {r2_score(y_val, preds):.4f}")
print(f"RMSE:     {mean_squared_error(y_val, preds)**0.5:.4f} °C")
print(f"MAE:      {mean_absolute_error(y_val, preds):.4f} °C")

# Optional: Predict reference parameters for the entire dataset (including invalid entries)
df['Predicted_Reference_Parameter'] = gbr_full.predict(X)

joblib.dump(gbr_full, 'rp_predictor.pkl') #saving the model

print("Model successfully saved to disk!")