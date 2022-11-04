#%%
#Imports
from project1 import df_trans
import sys
#Stationær
sys.path.insert(0, 'C:/Users/Mathias Damsgaard/Documents/GitHub/ML-Project')
from toolbox_02450 import rlr_validate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy.stats import boxcox
from scipy.io import loadmat

from sklearn import preprocessing, model_selection
import sklearn.linear_model as lm

#%%
# Define target and traning variables (without WindGustSpeed as per conclusion of last report)
var_reg = df_trans.drop(["WindGustSpeed", "RainToday"], axis=1)
var_clas = df_trans.drop(["WindGustSpeed", "Rainfall", "RainToday"], axis=1)
target_clas = df_trans["RainToday"]

# Standardize data
var_reg_scaled = preprocessing.scale(var_reg)
var_clas_scaled = preprocessing.scale(var_clas)

#%%
# Regression part a.2
X = var_reg_scaled[:,:8]
y = var_reg_scaled[:,8]
#attributeNames = list(var_reg.columns)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-4,5))

# Initialize variables
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M,K))
Error_lambda = np.empty(len(lambdas))

# Loop over lambda values
for i, l in enumerate(lambdas):
    k = 0

    # Loop for K-fold
    for train_index, test_index in CV.split(X,y):
        # Extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Estimate weights for the value of lambda, on entire training set
        lambdaI = l * np.eye(M)
        lambdaI[0,0] = 0 # Do not regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with lambda
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

        k += 1
    
    # Estimate generalization error
    Error_lambda[i] = np.sum(1/K * Error_test_rlr)

plt.plot(lambdas, Error_lambda, linestyle='-', marker='o', color='b')
plt.xscale('log')
plt.show()
