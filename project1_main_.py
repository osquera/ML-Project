import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from sklearn import decomposition


filename = 'Weather Training Data.csv'
df = pd.read_csv(filename)

df = df.loc[df['Location'] == 'Canberra']

df = df[["RainToday", "MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am", "Rainfall"]]


print(df)

print(df.isnull().sum())
# We remove all the places where RainToday is zero
df = df.dropna(subset=["RainToday"])

print(df.isnull().sum())

# We insert the mean on all NaN's in the dataset
for x in list(df.columns.values)[1:]:
    df[x] = df[x].fillna(df[x].mean())

print(df.isnull().sum())

# We turn Yes and No into binary
df.loc[df.RainToday == "Yes", "RainToday"] = 1
df.loc[df.RainToday == "No", "RainToday"] = 0

print(df.head())

# We turn the dataset into numpy array

X = df[["MinTemp", "MaxTemp", "WindGustSpeed", "Humidity9am", "Pressure9am",
        "Cloud9am", "Temp9am", "Rainfall"]].to_numpy()
print(X.shape)

#for x in ["MinTemp", "MaxTemp", "WindGustSpeed", "Humidity9am", "Pressure9am",
  #      "Cloud9am", "Temp9am", "Rainfall"]:
  #  idx = 0
   # plt.hist()



# Subtract mean value from data
Y = X - np.ones((2380, 1)) * X.mean(axis=0)

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold90 = 0.9
threshold95 = 0.95

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, 'x-')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold90, threshold90], 'k--')
plt.plot([1, len(rho)], [threshold95, threshold95], 'r--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual', 'Cumulative', 'Threshold 90', 'Threshold 95'])
plt.grid()
plt.show()


# We also want to do the correlation between the attributes
sns.displot(df, x="MinTemp", kde=True)
plt.title("Minimum temperature distribution")
plt.show()

sns.displot(df, x="MaxTemp", kde=True)
plt.title("Maximum temperature distribution", y=1.0, pad=-14)
plt.show()

sns.displot(df, x="WindGustSpeed", kde=True)
plt.title("Wind Gust Speed distribution")
plt.show()

sns.displot(df, x="Humidity9am", kde=True)
plt.title("Humidity at 9 am distribution")
plt.show()

sns.displot(df, x="Pressure9am", kde=True)
plt.title("Pressure at 9 am distribution")
plt.show()

sns.displot(df, x="Cloud9am", kde=True)
plt.title("Cloud level at 9 am distribution")
plt.show()

sns.displot(df, x="Temp9am", kde=True)
plt.title("Temperature at 9 am distribution")
plt.show()

sns.displot(df, x="Rainfall", kde=True)
plt.title("Rainfall during the day distribution")
plt.show()

sns.displot(df, x="Evaporation", kde=True)
plt.title("Evaporation distribution")
plt.show()

sns.displot(df, x="Sunshine", kde=True)
plt.title("Sunshine distribution")
plt.show()


#sns.pairplot(df[["MinTemp", "MaxTemp", "WindGustSpeed", "Humidity9am", "Pressure9am","Cloud9am", "Temp9am", "Rainfall"]])
#plt.show()

# We want to find the correlation
print(df[["MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
          "Cloud9am", "Temp9am", "Rainfall"]].corr())

sns.heatmap(df[["MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
                "Cloud9am", "Temp9am", "Rainfall"]].corr(), annot=True)
plt.xticks(rotation=45)
plt.show()


# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Project the centered data onto principal component space
Z = Y @ V

# Plot PCA of the data
f = plt.figure(figsize = (12,10))

location = 1
pca_num = 0
for x in range(1,5,1):
    for y in range(1,5,1):

        plt.subplot(4,4, location)
        if location == 1 or location == 6 or location == 11 or location == 16:
            plt.text(0.4,0.4,f'PC{x}',fontsize="xx-large")

        else:
            plt.plot(Z[:, x], Z[:, y], 'o', alpha=.5)

        location += 1

# Output result to screen
plt.show()
