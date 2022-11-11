# Imports
import sys
# sys.path.insert(0, 'C:/Users/Mathias Damsgaard/Documents/GitHub/ML-Project')
from toolbox_02450 import rlr_validate, train_neural_net
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid,plot,ylim)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy.stats import boxcox, t
from scipy.io import loadmat

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

# Define target and traning variables
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

model = lm.LogisticRegression()
model = model.fit(X,y_c)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_no_rain_prob = model.predict_proba(X)[:, 0]



# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y_c) / float(len(y_est))
true_pred = (np.sum(y_est == y_c) / float(len(y_est)))*100
# Display classification results

print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))
print('\npercentage of true prediction: {0:.3f}'.format(true_pred))
f = figure();
class0_ids = np.nonzero(y_c==0)[0].tolist()
plot(class0_ids, y_est_no_rain_prob[class0_ids], '.y')
class1_ids = np.nonzero(y_c==1)[0].tolist()
plot(class1_ids, y_est_no_rain_prob[class1_ids], '.r')
xlabel('Data object (Rain_sample)'); ylabel('Predicted prob. of no rain');
legend(['No rain', 'Rain'])
ylim(-0.01,1.5)

show()