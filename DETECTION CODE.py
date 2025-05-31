# Import necessary libraries
import pandas as pd
import numpy as np
import joblib

# Step 1: Load trained model, scaler, and label encoder
model_path = r"C:/PROJECT/MODEL/random_forestmain.pkl"
scaler_path = r"C:/PROJECT/MODEL/scaler.pkl"
label_encoder_path = r"C:/PROJECT/MODEL/label_encoder.pkl"

rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# Step 2: Load new dataset
new_data_path = r"C:/PROJECT/TEST/combine.csv"
new_data = pd.read_csv(new_data_path, low_memory=False)
new_data.columns = new_data.columns.str.strip()

# Step 3: Preprocessing
new_data_cleaned = new_data.dropna()
X_new = new_data_cleaned.apply(pd.to_numeric, errors='coerce')
X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
X_new = X_new.dropna()
new_data_cleaned = new_data_cleaned.loc[X_new.index]

# Step 4: Scaling
X_new_scaled = scaler.transform(X_new)

# Step 5: Prediction
predictions = rf_model.predict(X_new_scaled)

# Decode the class labels
predicted_labels = label_encoder.inverse_transform(predictions)

# Step 6: Add predictions to DataFrame
new_data_cleaned['Predicted_Label'] = predicted_labels

# Step 7: Save to CSV
output_path = r"D:/pythonProject6ISM/anomaly_predictions.csv"
new_data_cleaned.to_csv(output_path, index=False)
print(f"Multi-class prediction completed. Results saved to '{output_path}'.")
