# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 02:07:42 2019

@author: ibrahim mert oğurcu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


diabetes = datasets.load_diabetes()
    
print(diabetes.DESCR)

raw_data = diabetes.data
print(raw_data)

diabetes_df = pd.DataFrame(raw_data)
diabetes_df.head()

diabetes_df.columns = diabetes.feature_names
diabetes_df.head()


output_df = pd.DataFrame(diabetes.target, columns=["output"])
output_df.head()

all_df = pd.concat([diabetes_df, output_df], sort = False, axis = 1)
all_df.head()


X = diabetes_df['bmi'].values
Y = output_df.values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=9)
X_train = X_train.reshape(-1,1)
X_test= X_test.reshape(-1,1)

plt.scatter(X_train, y_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)


diabetes_y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
print('Kat sayı değeri:', model.coef_)
print("Hata oranı: %.2f" % mean_squared_error(y_test, diabetes_y_pred))
print('Varyans skoru (1 en iyi): %.2f' % r2_score(y_test, diabetes_y_pred))


plt.scatter(X_test, y_test,  c='g')
plt.plot(X_test, diabetes_y_pred, linewidth=2)
plt.show()

