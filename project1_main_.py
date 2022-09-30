import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
import sklearn.decomposition as idkman

filename = 'Weather Training Data.csv'
df = pd.read_csv(filename)

df = df.loc[df['Location'] == 'Canberra']

df = df[["RainToday", "MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am","Rainfall"]]

print(df)

print(df.isnull().sum())
#We remove all the places where RainToday is zero
df = df.dropna(subset=["RainToday"])

print(df.isnull().sum())

#We insert the mean on all NaN's in the dataset
for x in list(df.columns.values)[1:]:
    df[x] = df[x].fillna(df[x].mean())

print(df.isnull().sum())

#We turn Yes and No into binary
df.loc[df.RainToday == "Yes", "RainToday"] = 1
df.loc[df.RainToday == "No", "RainToday"] = 0

print(df.head())

#We turn the dataset into numpy array

X = df[["MinTemp", "MaxTemp", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am","Rainfall"]].to_numpy()
print(X.shape)

# Subtract mean value from data
Y = X - np.ones((2380,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold90 = 0.9
threshold95 = 0.95

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold90, threshold90],'k--')
plt.plot([1,len(rho)],[threshold95, threshold95],'r--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold 90', 'Threshold 95'])
plt.grid()
plt.show()

#We want to find the correlation
print(df[["MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am","Rainfall"]].corr())

#We also want to do the
sns.displot(df, x="MinTemp", hue="RainToday",kde=True)
plt.title("Minimum temperature distribution")
plt.show()


sns.pairplot(df[["MinTemp", "MaxTemp","WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am","Rainfall"]])
plt.show()
sns.heatmap(df[["MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am","Rainfall"]].corr(),annot=True)
plt.xticks(rotation=45)
plt.show()