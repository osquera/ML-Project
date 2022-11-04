import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from project2 import X,y,M



import sys

sys.path.insert(0, "C:/Users/Jacob pc/PycharmProject/ML-Project")
plt.rcParams.update({'font.size': 12})

# read XOR DATA from matlab datafile
#mat_data = loadmat('C:/Users/Jacob pc/Downloads/02450Toolbox_Python/Data/xor.mat')
#X = mat_data['X']
#y = mat_data['y']

#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
#classNames = [name[0] for name in mat_data['classNames'].squeeze()]
#N, M = X.shape
C = 8

# K-fold CrossValidation (4 folds here to speed up this example)
K = 10
CV = model_selection.KFold(K, shuffle=True)

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
n_hidden_units = 8  # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid()  # final tranfer function
)
# Since we're training a neural network for binary classification, we use a
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.CrossEntropyLoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
errors = []  # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the
# indices to be used for training and testing in each split, and calling
# the enumerate-method with this simply returns this indices along with
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)

    print('\n\tBest loss: {}\n'.format(final_loss))

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)  # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8)  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
    errors.append(error_rate)  # store error rate for current CV fold