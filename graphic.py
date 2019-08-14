import joblib
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

model_LR = joblib.load('model_LR.joblib')  
model_RF = joblib.load('model_RF.joblib')
model_GB = joblib.load('model_GB.joblib')

df = pd.read_csv('data_clean.csv')
# print(df.columns.values)

'''
'Unnamed: 0' 'price' 'bedrooms' 'bathrooms' 'sqft_living' 'sqft_lot'
 'floors' 'waterfront' 'condition' 'grade' 'sqft_above' 'sqft_basement'
 'yr_built' 'yr_renovated' 'lat' 'long' 'sqft_living15' 'sqft_lot15'
'''

y = df['price']
del df['price']
df = df.drop('Unnamed: 0', 1)
x = df.values
y = y.values


sns.set(style='darkgrid')

# print(model_LR.score(x, y))


# finding y best fit line
df['price_pred'] = model_LR.predict(x)
# print(df)

# plotting
# plt.scatter(df['sqft_living'], df['price_pred'])
plt.plot(df['sqft_living'], y)

plt.show()


# plt.scatter(df['bedrooms'], df['price'])
# plt.plot()
# # sns.relplot(df['bedrooms'], df['price'])
# plt.show()

