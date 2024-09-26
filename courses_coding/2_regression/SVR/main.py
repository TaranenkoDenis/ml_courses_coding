import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def print_table(X, y):
    for x, y in np.concatenate((X, y), axis=1):
        print('{:<30}{}'.format(x, y))

dataset = pd.read_csv('courses_coding/2_regression/SVR/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset['Salary']
y = np.array(y).reshape(len(y), 1)

print('Initial data:')
print_table(X, y)

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

print('Transformed data:')
print_table(X, y)

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

predicted_salary = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print('Predcited Salary: {}'.format(predicted_salary))

inversed_x = sc_X.inverse_transform(X)
inversed_y = sc_y.inverse_transform(y)

X_grid = np.arange(min(inversed_x), max(inversed_x), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
predicted_y = regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)
predicted_reversed_y = sc_y.inverse_transform(predicted_y)

plt.scatter(inversed_x, inversed_y, color='red', label='Data')
plt.plot(X_grid, predicted_reversed_y, color='blue', label='Predicted data')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
