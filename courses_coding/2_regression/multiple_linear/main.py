import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Notes
# In Multiply learning regression, we have no need to apply the Feature scaling.

dataset = pd.read_csv('courses_coding/2_regression/multiple_linear_regression/50_Startups.csv')
X = dataset.drop('Profit', axis=1).values
y = dataset['Profit']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)    # set the precision of the output

# To display the results as a vertical vectors, I used the reshape method
print(np.concatenate((y_pred.reshape(len(y_pred), 1), np.array(y_test).reshape(len(y_test), 1)), axis=1))

# Make chart to display the predicted values with the actual values
x_axis = np.arange(len(y_test))
plt.scatter(x_axis, y_test, color='red', label='Test data')
plt.scatter(x_axis, y_pred, color='green', label='Predicted test data')
plt.title('Predicted vs Actual profit')
plt.xlabel('Sturtup index')
plt.ylabel('Profit')
plt.legend()
plt.show()
