import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split


df = pd.read_csv('homeprices.csv')

X = df[["date"]]
Y = df[["food", "health", "other"]]


reg = linear_model.LinearRegression()
reg.fit(X, Y)

# pickle.dump(reg, open("model.pkl", "wb"))
print(reg.predict([[3000]]))
