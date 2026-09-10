import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    roc_auc_score
)
import joblib

# 1. Load the dataset
file_path = 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx'
df = pd.read_excel(file_path, sheet_name='Training_Data')

# 2. Convert Validity_Label to binary (1 = Valid, 0 = Invalid)
df['Validity_Binary'] = df['Validity_Label'].map({'Valid': 1, 'Invalid': 0})

# 3. Handle missing values
df['Sensor_S1'] = df['Sensor_S1'].fillna(-999)
df['Sensor_S2'] = df['Sensor_S2'].fillna(-999)
df['Sensor_S3'] = df['Sensor_S3'].fillna(-999)
df['Sensor_S4'] = df['Sensor_S4'].fillna(df['Sensor_S4'].median())

# 4. Prepare features and target
X = df.drop(columns=['Test_ID', 'Validity_Label', 'Validity_Binary', 'Reference_Parameter'])
y = df['Validity_Binary']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# 6. Evaluate Performance Parameters
val_probabilities = rf_classifier.predict_proba(X_val)[:, 1] # Probabilities for ROC-AUC
custom_threshold = 0.75
val_predictions = (val_probabilities >= custom_threshold).astype(int)




# --- Output Metrics ---
print("--- Model Performance Metrics ---")
print(f"Accuracy:      {accuracy_score(y_val, val_predictions):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_val, val_probabilities):.4f}\n")

print("Classification Report:")
print(classification_report(y_val, val_predictions, target_names=['Invalid (0)', 'Valid (1)']))

print("Confusion Matrix (Actual vs Predicted):")
cm = confusion_matrix(y_val, val_predictions)
print(f"True Negatives (Correct Invalid): {cm[0][0]} | False Positives (Missed Invalid): {cm[0][1]}")
print(f"False Negatives (Missed Valid):   {cm[1][0]} | True Positives (Correct Valid):   {cm[1][1]}")


joblib.dump(rf_classifier, 'validity_classifier.pkl') #saving the model

print("Model successfully saved to disk!")