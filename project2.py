#%%
#Imports
from project1 import df_trans
import sys
#Stationær
#sys.path.insert(0, 'C:/Users/Mathias Damsgaard/Documents/GitHub/ML-Project')
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
from sklearn import model_selection
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

X = var_reg_scaled[:,:8]
y = var_reg_scaled[:,8]
#attributeNames = list(var_reg.columns)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1

#%%
# Regression part a.2

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
N_k = np.empty((len(lambdas),K))

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

        #Hvorfor ikke også y?
        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Estimate weights for the value of lambda, on entire training set
        lambdaI = l * np.eye(M)
        lambdaI[0,0] = 0 # Do not regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with lambda
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
        N_k[i, k] = len(y_test)
        k += 1

    # Estimate generalization error
    Error_lambda[i] = np.sum(N_k[i,:]/N * Error_test_rlr)

plt.plot(lambdas, Error_lambda, linestyle='-', marker='o', color='b')
plt.xscale('log')
plt.show()

#%%
# Regression part b.1

# Create crossvalidation for split of data
K1 = 10
CV1 = model_selection.KFold(K1, shuffle=True)

K2 = 10
CV2 = model_selection.KFold(K2, shuffle=True)

lambdas = np.power(10.,range(-4,5))
k1 = 0
for par_idx, test_idx in CV1.split(X,y):
    X_par = X[par_idx]
    y_par = y[par_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    baseline = np.empty(K2)
    base_val_err = np.empty(K2)
    rlr_lambda_value = np.empty(K2)
    rlr_val_err = np.empty(K2)

    k2 = 0
    for train_idx, val_idx in CV2.split(X_par,y_par):
        X_train = X_par[train_idx]
        y_train = y_par[train_idx]
        X_val = X_par[val_idx]
        y_val = y_par[val_idx]

        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_val[:, 1:] = (X_val[:, 1:] - np.mean(X_val[:, 1:], 0)) / np.std(X_val[:, 1:], 0)

        baseline = np.mean(y_train)
        base_val_err[k2] = np.sum((y_val-baseline)**2)/len(y_val)

        for l in lambdas:
            Xty = X_train.T @ y_train
            XtX = X_train.T @ X_train
            lambdaI = l * np.eye(M)
            lambdaI[0,0] = 0
            w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            rlr_val_err[k2] = np.square(y_val-X_val @ w_rlr[:,k]).sum(axis=0)/y_val.shape[0]

        k2 += 1
    
    X_par[:, 1:] = (X_par[:, 1:] - np.mean(X_par[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
    X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

    k1 += 1

#%%
# Classification

# Our method 2 is ANN

# 1 We want to solve a binary classification

X = np.asarray(var_clas_scaled,dtype=float)
y = np.asarray(target_clas.values.tolist(),dtype=int)
attributeNames = [name for name in var_clas.columns.values.tolist()]
N, M = X.shape

# %%
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.float_power(10., np.arange(-5, 6, 0.5))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
                                                                                                      lambdas,
                                                                                                      internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        subplot(1, 2, 2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error', 'Validation error'])
        grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print(
    '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, -1], 2)))



