import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\...\\Venture_Capital_Startup_Investment_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
# We do this to change str into ints, such as the City in x
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# We did 'x = np.array(...)', to link this encoding into the 'past' x information we had.
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state= 0)

# Training the Multiple Linear Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2) # Present results with numerical values with only 2 decimals.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Display the 2 Vectors together

# Predicitng a new result with Multiple Linear Regression
print(regressor.predict(([[0.0, 0.0, 1.0, 165349.2, 136897.8, 471784.1]])))
# Here I added the above lines of code as when we want to analyze a specific x in order to know the predicted modeled result.
# 0.0, 0.0, 1.0 is New York in this casem, but you should verify yours, just in case.