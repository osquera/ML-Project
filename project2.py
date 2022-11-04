#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from scipy.stats import boxcox
from project1 import df_trans
from sklearn import preprocessing

#Define target and traning variables (without WindGustSpeed as per conclusion of last report)
target_reg = df_trans["Rainfall"]
target_class = df_trans["RainToday"]
var = df_trans.drop(["WindGustSpeed","Rainfall", "RainToday"], axis=1)

#Standardize data
var_scaled = preprocessing.scale(var)