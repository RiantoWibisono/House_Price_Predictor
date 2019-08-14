import numpy as np 
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('kc_house_data.csv')

# print(df.head)
# print(df.shape) # (21613, 21)

# print(df.columns.values) 
''' 
['id' 'date' 'price' 'bedrooms' 'bathrooms' 'sqft_living' 'sqft_lot'
'floors' 'waterfront' 'view' 'condition' 'grade' 'sqft_above'
'sqft_basement' 'yr_built' 'yr_renovated' 'zipcode' 'lat' 'long'
'sqft_living15' 'sqft_lot15']
'''
# print(df.isnull().any())        # no missing values

# =======================================
# Preprocessing data

# Removing unrelevant features / columns
df = df.drop(['id', 'date', 'zipcode', 'view'], 1)

# Removing outlier from associated columns
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
print(df.shape)     # (19770, 17)

# =======================================
# Separate variable into new dataframe from original dataframe which has only numerical values
df_corr = df.select_dtypes(include=[np.number])

# Correlation plot

corr = df_corr.corr()
print(corr)
plt.subplots(figsize = (8, 5))
sn.heatmap(corr, annot = True)
plt.title('Features Correlation Heat Map')

plt.subplots_adjust(left=.17, bottom=.24)


plt.show()

# # # Top 50% Correlation with sale price
# # top_feature = corr.index[abs(corr['price']>0.3)]
# # plt.subplots(figsize=(12, 8))
# # top_corr = df[top_feature].corr()
# # print(top_corr)
# # sn.heatmap(top_corr, annot=True)
# # plt.subplots_adjust(left=.17, bottom=.24)
# # plt.yticks(rotation = 0)
# # plt.show()


# # # Box plot
# # plt.figure(figsize=(18, 8))
# # sn.boxplot(df.grade, df.price)
# # plt.show()

# # =======================================
# # Finding most important features relative to target:
# # corr = df_corr.corr()
# # corr = corr.sort_values(['price'], ascending = False)
# # print(corr['price'])
# '''
# price            1.000000
# grade            0.620926
# sqft_living      0.608507
# sqft_living15    0.531485
# sqft_above       0.498569
# bathrooms        0.435438
# lat              0.417850
# bedrooms         0.286733
# floors           0.265269
# sqft_basement    0.253564
# yr_renovated     0.117815
# waterfront       0.107440
# sqft_lot15       0.073916
# sqft_lot         0.072210
# condition        0.059216
# yr_built         0.008372
# long            -0.002331
# '''

# # df = df.drop(['long', 'yr_built'], 1)
# # =======================================
# # Preparing data for prediction

# y = df['price']
# del df['price']

# # print(df.iloc[0])

# # format:
# # (bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, lat, long, sqft_living15, sqftlot15)
# zz = pd.Series([1, 1, 4000, 5000, 1, 0, 3, 3, 2000, 0, 1900, 2015, 47.5112, -122.2570, 4000, 5000])
# # print(zz.values)
# '''
# features:               max -- min       
# bedrooms                6 -- 1
# bathrooms               4.25 -- 0
# sqft_living             4460 -- 370
# sqft_lot                130680 -- 520
# floors                  3 -- 1
# waterfront              1 -- 0      --> view to the waterfront?
# condition               5 -- 1      --> overall condition?         # mean: 3
# grade                   11 -- 3     --> overall grade given to the housing unit, based on King County grading system   # man: 8 
# sqft_above              3860 -- 370--> square footage of house apart from basement
# sqft_basement           1510 -- 0   --> square footage of the basement
# yr_built                2015 -- 1900 
# yr_renovated            2015 -- 0
# lat                     47.7776 -- 47.1559
# long                    -121.315 -- -122.512
# sqft_living15           3721 -- 460 --> Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# sqft_lot15              55657 -- 651 --> lotSize area in 2015(implies-- some renovations)
# '''
# x = df.values
# y = y.values

# # Splitting data
# from sklearn.model_selection import train_test_split
# xtr, xts, ytr, yts = train_test_split(
#     x, 
#     y,
#     test_size = 0.1,
#     random_state = 1
# )

# # ==============================================================
# # Linear Regression
# # Training
# from sklearn import linear_model
# model = linear_model.LinearRegression()

# # Fit the model
# model.fit(xtr, ytr)

# # Prediction
# print("Predict value " + str(model.predict([zz.values])))
# # print("Real value " + str(yts[1500]))

# # Score/Accuracy
# print("Accuracy Linear Regression --> ", model.score(xts, yts)*100)

# # ==============================================================
# # Random Forest Regressor
# # Training
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state = 1)

# # Fit
# model.fit(xtr, ytr)

# # Prediction
# print("Predict value " + str(model.predict([zz.values])))
# # print("Real value " + str(yts[1500]))

# # Score/Accuracy
# print("Accuracy Random Forest Regressor --> ", model.score(xts, yts)*100)

# # ==============================================================
# # GradientBoostingRegressor

# # Training
# from sklearn.ensemble import GradientBoostingRegressor
# GBR = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state = 1)

# # Fit
# GBR.fit(xtr, ytr)

# # Prediction
# print("Predict value " + str(model.predict([zz.values])))
# # print("Real value " + str(yts[1500]))

# # Score/Accuracy
# print("Accuracy GradientBoostingRegressor --> ", GBR.score(xts, yts)*100)