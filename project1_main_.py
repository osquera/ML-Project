import numpy as np
import pandas as pd

filename = 'Weather Training Data.csv'
df = pd.read_csv(filename)

df = df.loc[df['Location'] == 'Canberra']

df = df[["RainToday", "MinTemp", "MaxTemp", "Evaporation", "Sunshine", "WindGustSpeed", "Humidity9am", "Pressure9am",
         "Cloud9am", "Temp9am"]]

print(df)

print(df.isnull().sum())



