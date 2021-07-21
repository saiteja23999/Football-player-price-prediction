import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pickle
dataframe = pd.read_csv("FOOTBALLL_DATASET.csv")
dataframe['fpl_sel'] = dataframe['fpl_sel'].astype('string')
dataframe['region'] = dataframe['region'].fillna(0)
for i in range(0,len(dataframe['fpl_sel'])):
  dataframe['fpl_sel'][i] = dataframe['fpl_sel'][i].strip("%")
dataframe['fpl_sel'] = dataframe['fpl_sel'].astype('float')
der_df = dataframe.drop(['name','club','position','nationality'],axis=1)
x= der_df.drop('market_value',axis=1)
y= der_df['market_value']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)


regressor = GradientBoostingRegressor(kernel='poly',gamma=1)
#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
result = model.score(xTest, yTest)
x = [28,1,4329,12,17.1,264,3,0,4,1,1,0]
result1 = model.predict([x])
print(result1)
