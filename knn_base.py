import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the spreadsheet data
file_path = 'MP.csv'
data = pd.read_csv(file_path)

# Ensure the necessary columns are present
required_columns = ['Height (cm)', 'Weight (kg)', 'Age (years)', 'BMI', 'Stress Fracture (%)']
if not all(col in data.columns for col in required_columns):
    raise ValueError("The spreadsheet must contain the following columns: " + ", ".join(required_columns))

# Separate features and target
X = data[['Height (cm)', 'Weight (kg)', 'Age (years)', 'BMI']]
y = data['Stress Fracture (%)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Define a function for new data prediction
def predict_stress_fracture(new_data):
    """
    Predicts stress fracture percentage for new input data.
    
    Parameters:
        new_data (DataFrame): New data with columns ['Height (cm)', 'Weight (kg)', 'Age (years)', 'BMI']
    
    Returns:
        ndarray: Predicted stress fracture percentages
    """
    new_data_scaled = scaler.transform(new_data)
    return knn.predict(new_data_scaled)

# Example usage with new data
new_data = pd.DataFrame({
    'Height (cm)': [170, 165],
    'Weight (kg)': [40, 70],
    'Age (years)': [18, 40],
    'BMI': [20.5, 25.7]
})
predictions = predict_stress_fracture(new_data)
print("Predicted Stress Fracture Percentages:", predictions)