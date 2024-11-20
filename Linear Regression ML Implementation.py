import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]

# Create a DataFrame
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
dataset = pd.DataFrame(boston_data, columns=columns)
dataset['Price'] = boston_target

# Display dataset overview
print("Dataset Head:")
print(dataset.head())
print("\nDataset Info:")
print(dataset.info())
print("\nDataset Description:")
print(dataset.describe())

# Check for missing values
if dataset.isnull().sum().sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values detected.")

# Correlation heatmap
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for visualization
sns.pairplot(dataset)
plt.show()

# Define independent and dependent variables
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
pickle.dump(scaler, open('scaling.pkl', 'wb'))

# Train the Linear Regression model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Print model coefficients and intercept
print("\nModel Coefficients:", regression.coef_)
print("Model Intercept:", regression.intercept_)

# Save the trained model
pickle.dump(regression, open('regmodel.pkl', 'wb'))
print("\nModel and scaler saved as 'regmodel.pkl' and 'scaling.pkl'.")

# Predict on the test set
reg_pred = regression.predict(X_test)

# Scatter plot of actual vs predicted values
plt.scatter(y_test, reg_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Residual analysis
residuals = y_test - reg_pred
sns.displot(residuals, kind="kde")
plt.title("Residual Distribution")
plt.show()

plt.scatter(reg_pred, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.show()

# Model evaluation metrics
mae = mean_absolute_error(y_test, reg_pred)
mse = mean_squared_error(y_test, reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, reg_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Adjusted R-squared: {adjusted_r2:.2f}")

# Predict on new data
new_data = X_test[0].reshape(1, -1)
scaled_new_data = scaler.transform(new_data)
predicted_price = regression.predict(scaled_new_data)
print("\nPrediction on New Data:", predicted_price)
