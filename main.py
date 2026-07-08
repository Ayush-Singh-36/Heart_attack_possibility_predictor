# importing all the libraries and dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# looking once at the data
data = pd.read_csv("C:\\Users\\ayush\\All of my AI development models\\heart_attack_possibility_predictor\\Heart_attack_possibility_predictor\\.venv\\data\\heart.csv")
data.info()

# dividing data into features and target sets
x = data.drop(columns=['target'])
y = data['target']

# dividing data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

# scaling data to fit for SVM model training
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# SVM model development
model = SVR(kernel = "rbf", C = 1.0, epsilon = 0.1)
model.fit(x_train_scaled, y_train)
print("Model training is complete")
predictions = model.predict(x_test_scaled)
print("Model is ready for predictions")
