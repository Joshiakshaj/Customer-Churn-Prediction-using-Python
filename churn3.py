# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# Load data
file_path = r'D:/Downloads/Churn_Prediction_Python_AWS_SageMaker-main/Churn_Prediction_Python_AWS_SageMaker-main/telco_data.xlsx'
df = pd.read_excel(file_path)

# Check if 'TotalCharges' column exists and handle it
if 'TotalCharges' in df.columns:
    # Handle missing or incorrect values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

# Handle other missing values
df = df.dropna()

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop(columns=['customerID']))

# Feature scaling
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Define features and target variable
X = df_encoded.drop(columns=['Churn_Yes', 'Churn_No'])
y = df_encoded['Churn_Yes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'criterion': ['gini', 'entropy']
}

# Train the model with the best parameters (assuming best_params are obtained)
best_model = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=10, criterion='gini', random_state=42)
best_model.fit(X_train_smote, y_train_smote)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train_smote, y_train_smote)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train_smote, y_train_smote)

# Predict and evaluate the tuned model
y_pred_best = best_model.predict(X_test)
print('Accuracy (tuned):', accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Get feature importances and sort them
importances = best_model.feature_importances_
features = X.columns
sorted_indices = importances.argsort()

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(features[sorted_indices], importances[sorted_indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Churn Prediction")
plt.show()

# Save the model to a .pkl file
file_path = 'D:/Downloads/ChurnProj/best_model.pkl'  # Specify the path where you want to save the .pkl file
with open(file_path, 'wb') as file:
    pickle.dump(best_model, file)
    print(f"Model saved as {file_path}")