import joblib
import pandas as pd
import joblib

def main():
    # 1. File path configuration
    file_path = 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx'
    test_sheet_name = 'Test_Data'
    output_filename = 'Whatever.csv'

    print("Loading test dataset...")
    df_test = pd.read_excel(file_path, sheet_name=test_sheet_name)

    # 2. Load the pre-trained models from disk
    # (Ensure these .pkl filenames match what you used when saving your models)
    print("Loading saved models...")
    validity_classifier = joblib.load('validity_classifier.pkl')
    rp_predictor = joblib.load('rp_predictor.pkl')

    # 3. Prepare features for the Classifier 
    # (Uses -999 extreme outlier imputation for S1, S2, S3 to isolate missing sensors as invalid)
    X_clf = df_test.drop(columns=['Test_ID']).copy()
    X_clf['Sensor_S1'] = X_clf['Sensor_S1'].fillna(-999)
    X_clf['Sensor_S2'] = X_clf['Sensor_S2'].fillna(-999)
    X_clf['Sensor_S3'] = X_clf['Sensor_S3'].fillna(-999)
    X_clf['Sensor_S4'] = X_clf['Sensor_S4'].fillna(X_clf['Sensor_S4'].median()) # Median imputation for decoy S4

    # Run Classifier Inference
    print("Running Validity Classification...")
    clf_preds = validity_classifier.predict(X_clf)
    # Map binary predictions back to string labels ('Valid' / 'Invalid')
    df_test['Validity_Label'] = ['Valid' if p == 1 else 'Invalid' for p in clf_preds]

    # 4. Prepare features for the Regressor
    # (Uses median imputation for S1, S2, S3 to avoid breaking regression calculations)
    reg_features = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 
                    'Test_Duration_min', 'Sensor_S1', 'Sensor_S2', 'Sensor_S3']
    X_reg = df_test[reg_features].copy()
    X_reg['Sensor_S1'] = X_reg['Sensor_S1'].fillna(X_reg['Sensor_S1'].median())
    X_reg['Sensor_S2'] = X_reg['Sensor_S2'].fillna(X_reg['Sensor_S2'].median())
    X_reg['Sensor_S3'] = X_reg['Sensor_S3'].fillna(X_reg['Sensor_S3'].median())

    # Run Regressor Inference
    print("Running Reference Parameter Prediction...")
    df_test['Predicted_Reference_Parameter'] = rp_predictor.predict(X_reg).round(6)

    # 5. Format final output structure (Test_ID, Predicted_Reference_Parameter, Validity_Label)
    # Matching the exact format expected by the evaluation pipeline
    submission_df = df_test[['Test_ID', 'Predicted_Reference_Parameter', 'Validity_Label']]

    # 6. Save to CSV
    submission_df.to_csv(output_filename, index=False)
    print(f"\nPipeline execution complete! Results successfully saved to '{output_filename}'.")
    print(submission_df.head())

if __name__ == '__main__':
    main()