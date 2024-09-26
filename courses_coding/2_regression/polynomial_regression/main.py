import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('courses_coding/2_regression/polynomial_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset['Salary']

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# plt.scatter(X, y, color='red', label='Actual Salaries')
# plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Regression')
# plt.plot(X, lin_reg_poly.predict(X_poly), color='green', label='Polynomial Regression')
# plt.legend()
# plt.title('Salary vs Position (Truth or Bluff)')
# plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_poly.predict(poly_reg.fit_transform([[6.5]])))
