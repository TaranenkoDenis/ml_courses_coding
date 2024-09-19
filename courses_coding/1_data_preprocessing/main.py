import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

def run():
    dataset = pd.read_csv('courses_coding/0_data_preparation/Data.csv')
    # feature data (independent variables), get all columns except the last one, -1 index of the last column
    X = dataset.iloc[:, :-1].values
    # predicted datag
    y = dataset.iloc[:, -1].values

    # Replace missing values with avg value for 2 and 3 columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

    # print('Feature data after replacing missing values with avg')
    # print('X = {}'.format(X))

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # print('Feature data after transforming first column by OneHotEncoder')
    # print('X = {}'.format(X))

    le = LabelEncoder()
    y = le.fit_transform(y)
    # print('Depended data after transforming by LabelEncoder')
    # print('y = {}'.format(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print('Train data')
    print('X_train = {}'.format(X_train))
    print('y_train = {}'.format(y_train))

    print('Test data')
    print('X_test = {}'.format(X_test))
    print('y_test = {}'.format(y_test))

    scaler = StandardScaler()

    # Cast again to numpy array to avoid error
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train[:, 3:] = scaler.fit_transform(X_train[:, 3: ])
    X_test[:, 3:] = scaler.transform(X_test[:, 3:])

    print('Train data after scaling')
    print('X_test = {}'.format(X_test))
    print('X_train = {}'.format(X_train))




if __name__ == "__main__":
    run()
