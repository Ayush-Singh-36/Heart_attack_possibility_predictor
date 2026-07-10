# importing all the libraries and dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle as pk

# looking once at the data
data = pd.read_csv("C:\\Users\\ayush\\All of my AI development models\\heart_attack_possibility_predictor\\Heart_attack_possibility_predictor\\data\\heart.csv")
data.info()

# dividing data into features and target sets
x = data.drop(columns=['target'])
y = data['target']

# dividing data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

# scaling data to fit for SVM model training
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# SVM model development
base_svm = SVC(kernel = "rbf", C = 1.0)
model = CalibratedClassifierCV(estimator = base_svm, ensemble = False)
model.fit(x_train_scaled, y_train)
print("Model training is complete")

# Making predictions out of the model
predictions = model.predict(x_test_scaled)
print("Model is ready for predictions")

# Using metrices to check predictions with actual values
accuracy = accuracy_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mrse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Accuracy Score: {accuracy * 100:.2f}% \nMean Absolute Error: {mae} \nMean Squared Error: {mse} \nMean Root Squared Error: {mrse} \nR2 Score: {r2 * 100:.2f}%")

# Create a dictionary of the artifacts you need for deployment
model_artifacts = {
    "scaler": scaler,
    "model": model,
    "metrics": {
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "mrse": mrse,
        "r2": r2
    }
}

# Define the file path to save your artifacts
file_path = "C:\\Users\\ayush\\All of my AI development models\\heart_attack_possibility_predictor\\Heart_attack_possibility_predictor\\heart_attack_model_artifacts.pkl"

# Write the artifacts to a file using pickle
with open(file_path, "wb") as file:
    pk.dump(model_artifacts, file)

print(f"Model artifacts successfully saved to {file_path}")