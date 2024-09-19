import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('courses_coding/2_regression/simple_linear_regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red', label='Train data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted train data')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted test data')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Single prediction for a single point with 12 years of experience.
# the "predict" method always expects a 2D array as the format of its inputs.
print(regressor.predict([[12]]))
