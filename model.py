# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset from CSV file
data = pd.read_csv('Iris.csv')

# Split data into features and target
X = data.drop(['Species','Id'], axis=1)
y = data['Species']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")

# Define the model filename
model_filename = 'random_forest_model.pkl'

# Check if the model file already exists
if os.path.exists(model_filename):
    print(f"{model_filename} already exists and will be replaced.")

# Save the model to a .pkl file (this will overwrite the existing file)
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved as {model_filename}")
