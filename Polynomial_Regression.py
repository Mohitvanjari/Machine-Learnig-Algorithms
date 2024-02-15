import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create sample dataset
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sales = np.array([10, 20, 30, 50, 80, 120, 150, 180, 200, 220])

# Fit polynomial curve to the data
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(months.reshape(-1, 1))
lin_reg = LinearRegression()
lin_reg.fit(X_poly, sales)

# Make predictions for the next 3 months
future_months = np.array([11, 12, 13])
future_X_poly = poly_reg.fit_transform(future_months.reshape(-1, 1))
future_sales = lin_reg.predict(future_X_poly)
print(future_sales)

fig = go.Figure()
fig.add_trace(go.Scatter(x=months, y=sales, name='Actual Sales'))
fig.add_trace(go.Scatter(x=months, y=lin_reg.predict(X_poly), name='Fitted Curve'))
fig.add_trace(go.Scatter(x=future_months, y=future_sales, name='Predicted Sales'))
fig.show()