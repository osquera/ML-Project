import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from sklearn import preprocessing, model_selection


import sys

#sys.path.insert(0, "C:/Users/Jacob pc/PycharmProject/ML-Project")
#plt.rcParams.update({'font.size': 12})


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

X = var_scaled
y = np.asarray(target_reg.values.tolist(),dtype=int)
#attributeNames = list(var_reg.columns)
N, M = X.shape

length = len(y)

y = y.reshape(length,1)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1
#%%

# K-fold CrossValidation (4 folds here to speed up this example)
K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=420)

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10, 10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K)))
subplot_size_2 = int(np.ceil(K / subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units1 = 1  # number of hidden units in the signle hidden layer
n_hidden_units2 = 5
n_hidden_units3 = 10

model1 = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units1),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units1, n_hidden_units1),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units1, n_hidden_units1),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units1, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid()  # final tranfer function
)

model2 = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units2),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units2, n_hidden_units2),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units2, n_hidden_units2),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units2, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid()  # final tranfer function
)

model3 = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units3),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units3, n_hidden_units3),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units3, n_hidden_units3),
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units3, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid()  # final tranfer function
)

loss_fn = torch.nn.MSELoss()

max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model1())))

tolerance = 1e-10

# Do cross-validation:
errors1 = []  # make a list for storing generalizaition error in each loop
errors2 = []
errors3 = []

############################ --- 1 --- ###############################

for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print(f'Model 1 with {n_hidden_units1} hidden units: '+'\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    net1, final_loss1, learning_curve1 = train_neural_net(model1,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=1,
                                                       max_iter=max_iter,
                                                       tolerance=tolerance)

    print('\n\tBest loss: {}\n'.format(final_loss1))

    # Determine estimated class labels for test set
    y_sigmoid1 = net1(X_test)  # activation of final note, i.e. prediction of network
    y_test_est1 = (y_sigmoid1 > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e1 = (y_test_est1 != y_test)
    error_rate1 = (sum(e1).type(torch.uint8) / len(y_test)).data.numpy()
    errors1.append(error_rate1)  # store error rate for current CV fold


############################ --- 2 --- ###############################


for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print(f'Model 2 with {n_hidden_units2} hidden units: '+'\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
    net2, final_loss2, learning_curve2 = train_neural_net(model2,
                                                          loss_fn,
                                                          X=X_train,
                                                          y=y_train,
                                                          n_replicates=1,
                                                          max_iter=max_iter,
                                                          tolerance=tolerance)



    print('\n\tBest loss: {}\n'.format(final_loss2))

    # Determine estimated class labels for test set
    y_sigmoid2 = net2(X_test)  # activation of final note, i.e. prediction of network
    y_test_est2 = (y_sigmoid2 > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e2 = (y_test_est2 != y_test)
    error_rate2 = (sum(e2).type(torch.uint8) / len(y_test)).data.numpy()
    errors2.append(error_rate2)  # store error rate for current CV fold




############################ --- 3 --- ###############################





for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print(f'Model 3 with {n_hidden_units3} hidden units: '+'\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
    net3, final_loss3, learning_curve3 = train_neural_net(model3,
                                                          loss_fn,
                                                          X=X_train,
                                                          y=y_train,
                                                          n_replicates=1,
                                                          max_iter=max_iter,
                                                          tolerance=tolerance)

    print('\n\tBest loss: {}\n'.format(final_loss3))

    # Determine estimated class labels for test set
    y_sigmoid3 = net3(X_test)  # activation of final note, i.e. prediction of network
    y_test_est3 = (y_sigmoid3 > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e3 = (y_test_est3 != y_test)
    error_rate3 = (sum(e3).type(torch.uint8) / len(y_test)).data.numpy()
    errors3.append(error_rate3)  # store error rate for current CV fold




#%%

print()