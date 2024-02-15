from sklearn import datasets
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Ridge Regression Implementation
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression Implementation
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Evaluate the models
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print("Ridge Regression MSE:", mse_ridge)
print("Lasso Regression MSE:", mse_lasso)

import plotly.express as px
import pandas as pd

# Create a DataFrame to store the results
results = pd.DataFrame({'Actual': y_test, 'Predicted_Ridge': y_pred_ridge, 
                        'Predicted_Lasso': y_pred_lasso})

# visualize the actual vs. predicted values for Ridge Regression
fig_ridge = px.scatter(results, x='Actual', y='Predicted_Ridge', 
                       title='Ridge Regression: Actual vs. Predicted',
                       labels={'Actual': 'Actual Values', 
                               'Predicted_Ridge': 'Predicted Values'},
                       trendline='ols')

# visualize the actual vs. predicted values for Lasso Regression
fig_lasso = px.scatter(results, x='Actual', y='Predicted_Lasso', 
                       title='Lasso Regression: Actual vs. Predicted',
                       labels={'Actual': 'Actual Values', 
                               'Predicted_Lasso': 'Predicted Values'},
                       trendline='ols')

fig_ridge.show()
fig_lasso.show()