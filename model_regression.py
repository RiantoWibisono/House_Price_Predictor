import numpy as np 
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('kc_house_data.csv')
df = df.drop(['id', 'date', 'zipcode', 'view'], 1)
df = df[(np.abs(stats.zscore(df[['price']])) < 3)]     
df = df[(np.abs(stats.zscore(df[['bedrooms']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['bathrooms']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_living']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_lot']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['floors']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_above']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_basement']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_living15']])) < 3)] 
df = df[(np.abs(stats.zscore(df[['sqft_lot15']])) < 3)] 

y = df['price']
del df['price']


x = df.values
y = y.values

print(x)
print(x.shape)
print(y)
print(y.shape)

from sklearn import linear_model
model_LR = linear_model.LinearRegression()
model_LR.fit(x, y)
print(model_LR.score(x,y))

from sklearn.ensemble import RandomForestRegressor
model_RF = RandomForestRegressor(n_estimators=100, random_state = 1)
model_RF.fit(x, y)

from sklearn.ensemble import GradientBoostingRegressor
model_GB = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state = 1)
model_GB.fit(x, y)

import joblib
joblib.dump(model_LR, 'model_LR.joblib')
joblib.dump(model_RF, 'model_RF.joblib')
joblib.dump(model_GB, 'model_GB.joblib')