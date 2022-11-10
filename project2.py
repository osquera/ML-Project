#%%
#Imports
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
from scipy.stats import boxcox, t
from scipy.io import loadmat

from sklearn import preprocessing, model_selection
import sklearn.linear_model as lm

#%%
# Loading data
filename = 'Weather Training Data.csv'
df = pd.read_csv(filename)

df = df.loc[df['Location'] == 'Sydney']

# Define variables (without WindGustSpeed as per conclusion of last report)
df = df[["RainToday", "MinTemp", "MaxTemp", "Evaporation", "Sunshine", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am", "Rainfall"]]

# We remove all the places where RainToday is zero
df = df.dropna(subset=["RainToday"])

# We insert the mean on all NaN's in the dataset
for x in list(df.columns.values)[1:]:
    df[x] = df[x].fillna(df[x].mean())

# We turn Yes and No into binary
df.loc[df.RainToday == "Yes", "RainToday"] = 1
df.loc[df.RainToday == "No", "RainToday"] = 0

# We transform by the following operations:
df_trans = df.copy()
df_trans['Humidity9am'] = df_trans['Humidity9am'].transform(np.sqrt)
df_trans['Evaporation'] = df_trans['Evaporation'].transform(np.sqrt)
df_trans['MaxTemp'] = df_trans['MaxTemp'].transform(np.log)

# Define target and traning variables
var = df_trans.drop(["RainToday", "Rainfall"], axis=1)
target_clas = df_trans["RainToday"]
target_reg = df_trans["Rainfall"]

# Standardize data
var_scaled = preprocessing.scale(var)

#%%
# Regression part a.2
# Define data
X = var_scaled
y = np.array(target_reg)
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
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X,y,lambdas,K)
# Display the results for the last cross-validation fold
figure(9, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
show()
#%%
# Regression part b.1

# Create crossvalidation for split of data
K1 = 10
CV1 = model_selection.KFold(K1, shuffle=True)

K2 = 10
CV2 = model_selection.KFold(K2, shuffle=True)

N, M = X.shape

k1 = 0
N_par = np.empty(K1)
N_k1 = np.empty(K1)
lambdas = np.power(10.,range(-4,5))
base_par = np.empty(K1)
base_gen_hat = np.empty(K1)
lambda_gen_hat = np.empty((K1,len(lambdas)))
base_test_err = np.empty(K1)
lambda_test_err = np.empty(K1)
w_rlr_par = np.empty((M,K1))
lambda_gen = np.empty(K1)

for par_idx, test_idx in CV1.split(X,y):
    X_par = X[par_idx]
    y_par = y[par_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    N_par[k1] = len(y_par)
    N_k1[k1] = len(y_test)

    k2 = 0
    N_k2 = np.empty(K2)

    base_train = np.empty(K2)
    base_val_err = np.empty(K2)
    rlr_lambda_value = np.empty(K2)
    rlr_val_err = np.empty(K2)

    w_rlr = np.empty((M,K2,len(lambdas)))
    lambda_val_err = np.empty((K2,len(lambdas)))
    y = y.squeeze()

    for train_idx, val_idx in CV2.split(X_par,y_par):
        X_train = X_par[train_idx]
        y_train = y_par[train_idx]
        X_val = X_par[val_idx]
        y_val = y_par[val_idx]
        
        N_k2[k2] = len(y_val)

        # Standardize the training and set based on the partial set
        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_val[:, 1:] = (X_val[:, 1:] - np.mean(X_val[:, 1:], 0)) / np.std(X_val[:, 1:], 0)

        # Baseline
        base_train[k2] = np.mean(y_train)
        base_val_err[k2] = np.sum((y_val-base_train[k2])**2)/len(y_val)

        # Regularization
        # Precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w_rlr[:,k2,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate validation error
            lambda_val_err[k2,l] = np.power(y_val-X_val @ w_rlr[:,k2,l].T,2).mean(axis=0)

        k2 += 1
    
    # Estimate generalization error
    base_gen_hat[k1] = np.sum(N_k2/N_par[k1] * base_val_err)
    lambda_gen_hat[k1] = np.sum(N_k2/N_par[k1] * lambda_val_err.T, axis=1)
    lambda_opt = lambdas[np.argmin(lambda_gen_hat[k1])]

    X_par[:, 1:] = (X_par[:, 1:] - np.mean(X_par[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
    X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

    # Baseline
    base_par[k1] = np.mean(y_par)
    base_test_err[k1] = np.sum((y_par-base_par[k1])**2)/len(y_val)

    # Regularization
    # Precompute terms
    Xty = X_par.T @ y_par
    XtX = X_par.T @ X_par
    
    # Compute parameters for current value of lambda and current CV fold
    lambdaI = lambda_opt * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    w_rlr_par[:,k1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Evaluate test error
    lambda_test_err[k1] = np.power(y_test-X_test @ w_rlr_par[:,k1].T,2).mean(axis=0)

    print(f"Baseline test error for fold {k1+1}: {base_test_err[k1]}")
    print(f"Lambda test error for fold {k1+1}: {lambda_test_err[k1]} with lambda: {lambda_opt}")

    k1 += 1

base_gen = np.sum(N_k1/N * base_test_err)
lambda_gen = np.sum(N_k1/N * lambda_test_err)
print(f"Baseline gen error: {base_gen}")
print(f"Lambda gen error: {lambda_gen}")

#%%
# Regression b3
rho = 1/K2
df = K2-1
alpha = 0.05

r_k1 = base_test_err - lambda_test_err
r_hat = np.mean(r_k1)
s_2 = np.var(r_k1)
sigma_hat = np.sqrt((1/K2+1/(K1-1))*s_2)
t_value = r_hat/(sigma_hat*np.sqrt(1/K2+1/(K1-1)))
p_value = 2*t.cdf(-abs(t_value),df=df)
print(f"Base vs rlr p-value: {p_value}")
ci = t.interval(confidence=1-alpha,df=df,loc=r_hat,scale=sigma_hat)
print(f"Base vs rlr CI: {ci}")

#%%
# Classification

# Our method 2 is ANN

# 1 We want to solve a binary classification

X = np.asarray(var_scaled,dtype=float)
y = np.array(target_clas.values)
attributeNames = [name for name in var.columns.values.tolist()]
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



