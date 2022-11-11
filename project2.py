# %%
# Imports
import sys
# sys.path.insert(0, 'C:/Users/Mathias Damsgaard/Documents/GitHub/ML-Project')
from toolbox_02450 import rlr_validate, train_neural_net
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox, t

from sklearn import preprocessing, model_selection
import sklearn.linear_model as lm

import torch

# %%
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

# Define target and training variables
var = df_trans.drop(["RainToday", "Rainfall"], axis=1)
target_clas = df_trans["RainToday"]
target_reg = df_trans["Rainfall"]

# Standardize data
var_scaled = preprocessing.scale(var)

# Define data
X = var_scaled
y_r = np.asarray(target_reg.values.tolist(), dtype=int)
y_c = np.asarray(target_clas.values.tolist(), dtype=int)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
# attributeNames = [u'Offset']+attributeNames
M = M + 1

# %%
# Regression part a.2

# Create cross-validation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10., range(-4, 5))

# Initialize variables
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
w_rlr = np.empty((M, K))
Error_lambda = np.empty(len(lambdas))
N_k = np.empty((len(lambdas), K))

# Loop over lambda values
for i, l in enumerate(lambdas):
    k = 0

    # Loop for K-fold
    for train_index, test_index in CV.split(X, y_r):
        # Extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y_r[train_index]
        X_test = X[test_index]
        y_test = y_r[test_index]

        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Estimate weights for the value of lambda, on entire training set
        lambdaI = l * np.eye(M)
        lambdaI[0, 0] = 0  # Do not regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with lambda
        Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
        N_k[i, k] = len(y_test)
        k += 1

    # Estimate generalization error
    Error_lambda[i] = np.sum(N_k[i, :] / N * Error_test_rlr)

plt.plot(lambdas, Error_lambda, linestyle='-', marker='o', color='b')
plt.xscale('log')
plt.show()

# %%
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y_r, lambdas, K)
# Display the results for the last cross-validation fold
figure(9, figsize=(12, 8))
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
ylabel('Squared error (cross-validation)')
legend(['Train error', 'Validation error'])
grid()
show()
# %%
# Regression part b.1
# Create cross-validation for split of data
K1 = 10
CV1 = model_selection.KFold(K1, shuffle=True, random_state=69)

K2 = 10
CV2 = model_selection.KFold(K2, shuffle=True, random_state=420)

# Define the model structure
n_units = [1, 3, 5]  # number of hidden units in the single hidden layer
loss_fn = torch.nn.MSELoss()
max_iter = 10000
tolerance = 1e-10

lambdas = np.power(10., range(-4, 9))
lambda_gen_hat = np.empty((K1, len(lambdas)))
nn_gen_hat = np.empty((K1, len(n_units)))
base_test_err = np.empty(K1)
lambda_test_err = np.empty(K1)
nn_test_err = np.empty(K1)
w_rlr_par = np.empty((M, K1))
lambda_gen = np.empty(K1)

k1 = 0
for par_idx, test_idx in CV1.split(X, y_r):
    print('Outer cross-validation fold: {0}/{1}'.format(k1 + 1, K1))
    X_par = X[par_idx]
    y_par = y_r[par_idx]
    X_test = X[test_idx]
    y_test = y_r[test_idx]

    rlr_lambda_value = np.empty(K2)
    rlr_val_err = np.empty(K2)
    w_rlr = np.empty((M, K2, len(lambdas)))
    lambda_val_err = np.empty((K2, len(lambdas)))
    nn_val_err = np.empty((K2, len(n_units)))

    k2 = 0
    for train_idx, val_idx in CV2.split(X_par, y_par):
        print('Inner cross-validation fold: {0}/{1}'.format(k2 + 1, K2))
        X_train = X_par[train_idx]
        y_train = y_par[train_idx]
        X_val = X_par[val_idx]
        y_val = y_par[val_idx]

        # Standardize the training and set based on the partial set
        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_val[:, 1:] = (X_val[:, 1:] - np.mean(X_val[:, 1:], 0)) / np.std(X_val[:, 1:], 0)

        # Regularization
        # Precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w_rlr[:, k2, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate validation error
            lambda_val_err[k2, l] = np.power(y_val - X_val @ w_rlr[:, k2, l].T, 2).mean(axis=0)

        y_nn = y_train.reshape(len(y_train), 1)

        for n, h in enumerate(n_units):
            model_r = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),  # M features to H hidden units
                # 1st transfer function, either Tanh or ReLU:
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, 1),  # H hidden units to 1 output neuron
            )

            print(f'Model with {h} hidden units')

            # Extract training and test set for current CV fold,
            # and convert them to PyTorch tensors
            X_train_nn = torch.Tensor(X_train)
            y_train_nn = torch.Tensor(y_nn)
            X_test_nn = torch.Tensor(X_val)
            y_test_nn = torch.Tensor(y_val)

            net, final_loss, learning_curve = train_neural_net(model_r,
                                                               loss_fn,
                                                               X=X_train_nn,
                                                               y=y_train_nn,
                                                               n_replicates=1,
                                                               max_iter=max_iter,
                                                               tolerance=tolerance)

            # Determine estimated class labels for test set
            y_test_est = net(X_test_nn)  # activation of final note, i.e. prediction of network
            # Determine validation error and error rate
            nn_val_err[k2, n] = sum(((y_test_est.float() - y_test_nn.float()) ** 2)[0]) / len(y_test_nn)

        k2 += 1

    # Estimate generalization error
    lambda_gen_hat[k1] = np.sum(1 / K2 * lambda_val_err, axis=0)
    lambda_opt = lambdas[np.argmin(lambda_gen_hat[k1])]
    nn_gen_hat[k1] = np.sum(1 / K2 * nn_val_err, axis=0)
    unit_opt = n_units[np.argmin(nn_gen_hat[k1])]

    X_par[:, 1:] = (X_par[:, 1:] - np.mean(X_par[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
    X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

    # Baseline
    baseline = np.mean(y_par)
    base_test_err[k1] = np.sum((y_par - baseline) ** 2) / len(y_val)

    # Regularization
    # Precompute terms
    Xty = X_par.T @ y_par
    XtX = X_par.T @ X_par

    # Compute parameters for current value of lambda and current CV fold
    lambdaI = lambda_opt * np.eye(M)
    lambdaI[0, 0] = 0  # remove bias regularization
    w_rlr_par[:, k1] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Evaluate test error
    lambda_test_err[k1] = np.power(y_test - X_test @ w_rlr_par[:, k1].T, 2).mean(axis=0)

    # Neural network
    y_nn = y_par.reshape(len(y_par), 1)

    model_r = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, unit_opt),  # M features to H hidden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, unit_opt),
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, unit_opt),
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, 1),  # H hidden units to 1 output neuron
    )

    print(f'Optimal model with {unit_opt} hidden units')

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train_nn = torch.Tensor(X_par)
    y_train_nn = torch.Tensor(y_nn)
    X_test_nn = torch.Tensor(X_test)
    y_test_nn = torch.Tensor(y_test)

    net, final_loss, learning_curve = train_neural_net(model_r,
                                                       loss_fn,
                                                       X=X_train_nn,
                                                       y=y_train_nn,
                                                       n_replicates=1,
                                                       max_iter=max_iter,
                                                       tolerance=tolerance)

    # Determine estimated class labels for test set
    y_test_est = net(X_test_nn)  # activation of final note, i.e. prediction of network
    # Determine validation error and error rate
    nn_test_err[k1] = sum(((y_test_est.float() - y_test_nn.float()) ** 2)[0]) / len(y_test_nn)

    print(f"Baseline test error for fold {k1 + 1}: {base_test_err[k1]}")
    print(f"RLR test error for fold {k1 + 1}: {lambda_test_err[k1]} with lambda: {1 / lambda_opt}")
    print(f"NN test error for fold {k1 + 1}: {nn_test_err[k1]} with hidden units: {unit_opt}")

    k1 += 1

base_gen = np.sum(1 / K1 * base_test_err)
lambda_gen = np.sum(1 / K1 * lambda_test_err)
nn_gen = np.sum(1 / K1 * nn_test_err)
print(f"Baseline gen error: {base_gen}")
print(f"RLR gen error: {lambda_gen}")
print(f"NN gen error: {nn_gen}")
# %%
# Regression b3
df = K2 - 1
alpha = 0.05

r_k1 = np.empty((3, K1))
r_k1[0] = base_test_err - lambda_test_err
r_k1[1] = base_test_err - nn_test_err
r_k1[2] = lambda_test_err - nn_test_err
r_hat = np.mean(r_k1, axis=1)
s_2 = np.var(r_k1, axis=1)
sigma_hat = np.sqrt((1 / K2 + 1 / (K1 - 1)) * s_2)
t_value = r_hat / (sigma_hat * np.sqrt(1 / K2 + 1 / (K1 - 1)))

p_value_br = 2 * t.cdf(-abs(t_value[0]), df=df)
p_value_bn = 2 * t.cdf(-abs(t_value[1]), df=df)
p_value_rn = 2 * t.cdf(-abs(t_value[2]), df=df)

print(f"Base vs RLR p-value: {p_value_br}")
ci_br = t.interval(confidence=1 - alpha, df=df, loc=r_hat[0], scale=sigma_hat[0])
print(f"Base vs RLR CI: {ci_br}")

print(f"Base vs NN p-value: {p_value_bn}")
ci_bn = t.interval(confidence=1 - alpha, df=df, loc=r_hat[1], scale=sigma_hat[1])
print(f"Base vs NN CI: {ci_bn}")

print(f"RLR vs NN p-value: {p_value_rn}")
ci_rn = t.interval(confidence=1 - alpha, df=df, loc=r_hat[2], scale=sigma_hat[2])
print(f"RLR vs NN CI: {ci_rn}")
# %%
#############################################
############## Classification ###############
#############################################

# Our method 2 is ANN

# 1 We want to solve a binary classification

X = var_scaled
y_c = np.asarray(target_clas.values.tolist(), dtype=int)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
# attributeNames = [u'Offset']+attributeNames
M = M + 1

K1 = 10
CV1 = model_selection.KFold(K1, shuffle=True, random_state=69)

K2 = 10
CV2 = model_selection.KFold(K2, shuffle=True, random_state=420)

# Define the model structure
n_units = [1, 3, 5]  # number of hidden units in the single hidden layer
loss_fn = torch.nn.MSELoss()
max_iter = 10000
tolerance = 1e-10

lambdas = np.float_power(10., np.arange(0, 1.4, 0.2))
lambda_gen_hat = np.empty((K1, len(lambdas)))
nn_gen_hat = np.empty((K1, len(n_units)))
base_test_err = np.empty(K1)
lambda_test_err = np.empty(K1)
nn_test_err = np.empty(K1)
w_rlr_par = np.empty((M, K1))

k1 = 0
for par_idx, test_idx in CV1.split(X, y_c):
    print('Outer cross-validation fold: {0}/{1}'.format(k1 + 1, K1))
    X_par = X[par_idx]
    y_par = y_r[par_idx]
    X_test = X[test_idx]
    y_test = y_r[test_idx]

    rlr_lambda_value = np.empty(K2)
    rlr_val_err = np.empty(K2)
    w_rlr = np.empty((M, K2, len(lambdas)))
    lambda_val_err = np.empty((K2, len(lambdas)))
    nn_val_err = np.empty((K2, len(n_units)))

    k2 = 0
    for train_idx, val_idx in CV2.split(X_par, y_par):
        print('Inner cross-validation fold: {0}/{1}'.format(k2 + 1, K2))
        X_train = X_par[train_idx]
        y_train = y_par[train_idx]
        X_val = X_par[val_idx,]
        y_val = y_par[val_idx]

        # Standardize the training and set based on the partial set
        X_train[:, 1:] = (X_train[:, 1:] - np.mean(X_train[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
        X_val[:, 1:] = (X_val[:, 1:] - np.mean(X_val[:, 1:], 0)) / np.std(X_val[:, 1:], 0)

        # Regularization
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold

            model = lm.LogisticRegression(max_iter=100000, solver='saga', C=lambdas[l], penalty='l1')
            model = model.fit(X_train, y_train)
            y_est = model.predict(X_val)
            # Evaluate validation error
            lambda_val_err[k2, l] = np.sum(y_est != y_val) / len(y_est)

        y_train_nn = y_train.reshape(len(y_train), 1)  # Ændrer så den passer i rigtige format, ex: [[0][1][1]]
        y_val_nn = y_val.reshape(len(y_val), 1)

        for n, h in enumerate(n_units):
            model_c = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),  # M features to H hidden units
                # 1st transfer function, either Tanh or ReLU:
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),  # torch.nn.ReLU(),
                torch.nn.Linear(h, 1),  # H hidden units to 1 output neuron
                torch.nn.Sigmoid()  # final transfer function
            )

            print(f'Model with {h} hidden units')


            # Extract training and test set for current CV fold,
            # and convert them to PyTorch tensors
            X_train_nn = torch.Tensor(X_train)
            y_train_nn = torch.Tensor(y_train_nn)
            X_test_nn = torch.Tensor(X_val)
            y_test_nn = torch.Tensor(y_val_nn)

            net, final_loss, learning_curve = train_neural_net(model_c,
                                                               loss_fn,
                                                               X=X_train_nn,
                                                               y=y_train_nn,
                                                               n_replicates=1,
                                                               max_iter=max_iter,
                                                               tolerance=tolerance)

            # Determine estimated class labels for test set
            y_sigmoid = net(X_test_nn)
            y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
            y_test_nn = y_test_nn.type(dtype=torch.uint8)
            # Determine errors and error rate
            nn_val_err[k2, n] = (sum(y_test_est != y_test_nn).type(torch.uint8) / len(y_test)).data.numpy()

        k2 += 1

    # Estimate generalization error
    lambda_gen_hat[k1] = np.sum(1 / K2 * lambda_val_err, axis=0)
    lambda_opt = lambdas[np.argmin(lambda_gen_hat[k1])]
    nn_gen_hat[k1] = np.sum(1 / K2 * nn_val_err, axis=0)
    unit_opt = n_units[np.argmin(nn_gen_hat[k1])]

    X_par[:, 1:] = (X_par[:, 1:] - np.mean(X_par[:, 1:], 0)) / np.std(X_train[:, 1:], 0)
    X_test[:, 1:] = (X_test[:, 1:] - np.mean(X_test[:, 1:], 0)) / np.std(X_test[:, 1:], 0)

    # Baseline
    baseline = np.mean(y_par)
    base_test_err[k1] = np.sum((y_par - baseline) ** 2) / len(y_val)

    # Regularization
    # Compute parameters for current value of lambda and current CV fold

    model = lm.LogisticRegression(max_iter=100000, solver='saga', C=lambda_opt, penalty='l1')
    model = model.fit(X_par, y_par)
    y_est = model.predict(X_test)

    # Evaluate test error
    lambda_test_err[k1] = np.sum(y_est != y_test) / len(y_est)

    # Neural network
    y_par_nn = y_par.reshape(len(y_par), 1)
    y_test_nn = y_test.reshape(len(y_test), 1)

    model_c = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, unit_opt),  # M features to H hiden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, unit_opt),
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, unit_opt),
        torch.nn.Tanh(),  # torch.nn.ReLU(),
        torch.nn.Linear(unit_opt, 1),  # H hidden units to 1 output neuron
        torch.nn.Sigmoid()  # final tranfer function
    )

    print(f'Optimal model with {unit_opt} hidden units')

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train_nn = torch.Tensor(X_par)
    y_train_nn = torch.Tensor(y_par_nn)
    X_test_nn = torch.Tensor(X_test)
    y_test_nn = torch.Tensor(y_test_nn)

    net, final_loss, learning_curve = train_neural_net(model_c,
                                                       loss_fn,
                                                       X=X_train_nn,
                                                       y=y_train_nn,
                                                       n_replicates=1,
                                                       max_iter=max_iter,
                                                       tolerance=tolerance)

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test_nn)
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test_nn = y_test_nn.type(dtype=torch.uint8)
    # Determine errors and error rate
    nn_val_err[k2, n] = (sum(y_test_est != y_test_nn).type(torch.uint8) / len(y_test)).data.numpy()

    print(f"Baseline test error for fold {k1 + 1}: {base_test_err[k1]}")
    print(f"RLR test error for fold {k1 + 1}: {lambda_test_err[k1]} with lambda: {1 / lambda_opt}")
    print(f"NN test error for fold {k1 + 1}: {nn_test_err[k1]} with hidden units: {unit_opt}")

    k1 += 1

base_gen = np.sum(1 / K1 * base_test_err)
lambda_gen = np.sum(1 / K1 * lambda_test_err)
nn_gen = np.sum(1 / K1 * nn_test_err)
print(f"Baseline gen error: {base_gen}")
print(f"RLR gen error: {lambda_gen}")
print(f"NN gen error: {nn_gen}")
# %%
# Classification test
df = K2 - 1
alpha = 0.05

r_k1 = np.empty((3, K1))
r_k1[0] = base_test_err - lambda_test_err
r_k1[1] = base_test_err - nn_test_err
r_k1[2] = lambda_test_err - nn_test_err
r_hat = np.mean(r_k1, axis=1)
s_2 = np.var(r_k1, axis=1)
sigma_hat = np.sqrt((1 / K2 + 1 / (K1 - 1)) * s_2)
t_value = r_hat / (sigma_hat * np.sqrt(1 / K2 + 1 / (K1 - 1)))

p_value_br = 2 * t.cdf(-abs(t_value[0]), df=df)
p_value_bn = 2 * t.cdf(-abs(t_value[1]), df=df)
p_value_rn = 2 * t.cdf(-abs(t_value[2]), df=df)

print(f"Base vs RLR p-value: {p_value_br}")
ci_br = t.interval(confidence=1 - alpha, df=df, loc=r_hat[0], scale=sigma_hat[0])
print(f"Base vs RLR CI: {ci_br}")

print(f"Base vs NN p-value: {p_value_bn}")
ci_bn = t.interval(confidence=1 - alpha, df=df, loc=r_hat[1], scale=sigma_hat[1])
print(f"Base vs NN CI: {ci_bn}")

print(f"RLR vs NN p-value: {p_value_rn}")
ci_rn = t.interval(confidence=1 - alpha, df=df, loc=r_hat[2], scale=sigma_hat[2])
print(f"RLR vs NN CI: {ci_rn}")
