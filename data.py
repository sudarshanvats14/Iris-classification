import pandas as pd
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Data\\iris.csv')

columns_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
df.columns = columns_names

new_row = pd.Series([5.1, 3.5, 1.4, 0.2, 'Iris-setosa'], index=columns_names)
df = df.append(new_row, ignore_index=True)
print(df)

sns.countplot(data=df, x='Species')
plt.show()

print(df["Species"].unique())


np.random.seed(42)
tf.random.set_seed(42)


data = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(data)

x = data.iloc[:, :4].values
y = data.iloc[:, 4].values

print("Shape of X", x.shape)
print("Shape of Y", y.shape)
print("Examples of X\n", x[:3])
print("Examples of y\n", y[:3])

X_normalized = normalize(x, axis=0)
print("Examples of X_normalized\n", X_normalized[:3])

encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1)).toarray()
print("Examples of y_onehot\n", y_onehot[:3])
print(y_onehot.shape)

X_final = np.concatenate((X_normalized, y_onehot), axis=1)
print("Final Feature Matrix\n", X_final[:3])
