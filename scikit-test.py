# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from common import *

# Load the diabetes dataset
#diabetes = datasets.load_diabetes()


# Use only one feature
#diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
#diabetes_X_train = diabetes_X[:-20]
#diabetes_X_test = diabetes_X[-20:]


# Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]

import sys

X, y = load_dataset(sys.argv[1], sys.argv[2])
X = preprocess(X, 'gaussian')
X = np.asarray(X)
y = np.asarray(y)

N_train = 32000
N_test = 8000
B = 32

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[-N_test:]
y_test = y[-N_test:]

# Create linear regression object
regr = linear_model.SGDRegressor(alpha=0.0001, n_iter=10000)
np.random.shuffle(X_train)

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(X_test, y_test))
