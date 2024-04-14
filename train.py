import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import normalize
from data import df, X_normalized, y_onehot
'''
80% -- train data
20% -- test data
'''
total_length=len(df)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y_onehot[:train_length]
y_test=y_onehot[train_length:]
print(X_test)
print(X_test.shape)
print("shape of y train:",y_train.shape)
print("shape of y test:",y_test.shape)
print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])