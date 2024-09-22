# Stone Barnard
# Data Science Final Project
# 21 August 2024

#import pyreader as py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.inspection import permutation_importance


#import the data into a dataframe. I converted the .rdata file in R first but would recommend the use of pyreader.
df = pd.read_csv('carAd.csv')

#review the features of the dataset
#df.info()

#reset the index
df.reset_index(drop=True, inplace=True)

#select those features that lend themselves to rendom forest analysis

#goal is to use model, body type, fuel type, top six used colors, the number of seats, the number of doors.
carAd = df[['Genmodel','Color','Bodytype','Fuel_type','Seat_num','Door_num','Price']]

#print(carAd)

#Start cleaning the data

#print(carAd["Genmodel"].value_counts(dropna = False)) 

#list to select specific models
models = ['L200', 'Q3', 'CX-5', 'XC90']

#select our models from the Genmodel column
carAd = carAd.loc[carAd['Genmodel'].isin(models)]
 
#select the top 6 colors sold
carColors = carAd["Color"].value_counts().index.tolist()[:6]
carAd = carAd.loc[carAd['Color'].isin(carColors)]

#Just Diesel and Petrol
carAd = carAd[(carAd['Fuel_type'] == 'Petrol') | (carAd['Fuel_type'] == 'Diesel')]

#Just SUV and Pickup
carAd = carAd[(carAd['Bodytype'] == 'SUV') | (carAd['Bodytype'] == 'Pickup')]

#look at non-null count
#carAd.info()

#drop the Seat_num, Door_num null values
carAd.dropna(inplace=True)

#convert price to an integer
carAd['Price'] = pd.to_numeric(carAd['Price'], errors='coerce', downcast='integer')

#last check for anything odd
#print(carAd.loc[carAd['Price'].idxmax()])

#initial data analysis of each feature, print a histogram
#for columns in carAd.columns: 
#   plt.figure(figsize=(4,3))
#   sns.histplot(data=carAd,x=columns
#                , bins= 10)
#   plt.title(columns)
#   plt.show()

#Extract categorical columns from the dataframe
categorical_columns = carAd[['Bodytype','Color','Fuel_type','Genmodel']]
#print(categorical_columns)

encoded_columns = pd.get_dummies(categorical_columns, sparse= False, dtype='float')
#print(encoded_columns)

#join data sets

first_all_data = carAd.join(encoded_columns, how='inner')
clean_all_data = first_all_data.drop(columns = ['Bodytype', 'Color','Fuel_type','Genmodel'])

#print(clean_all_data)
#Initialize OneHotEncoder
#encoder = OneHotEncoder()
#clean_all_data.to_csv('final_csv.csv')

#Apply one-hot encoding to the categorical columns
#one_hot_encoded = encoder.fit_transform(carAd[categorical_columns])
#print(one_hot_encoded)

#transform the selected columns
#cTrans = make_column_transformer((encoder, categorical_columns), remainder= 'passthrough')

#set the model params
etr = ExtraTreesRegressor(random_state=17, max_features= None, verbose=2)

#one hot encode the columns. Pass in the complete dataset
#cTrans.fit(carAd_transform)



#create the training and testing data
x,testX,y,testY = train_test_split(clean_all_data, clean_all_data.iloc[:, 6], test_size = .1, stratify = None, random_state = 17)

#create the pipeline that takes in the encoded data and applies the model
#pipe = Pipeline(steps= [('ctrf', encoded_columns), ('model', etr)])

#train the model
#pipe.fit(x, y)

#Show the results
#print("R\N{SUPERSCRIPT TWO} =", pipe.score(x,y))
#R2 = 0.14278  FAIL


#trPr = pipe.predict(x)
#print("RMSE =",metrics.mean_squared_error(y,trPr,squared = False))
#RMSE = 9058.74

#evaluating testing performance
#pred = pipe.predict(testX)
#print("R\N{SUPERSCRIPT TWO} =",metrics.explained_variance_score(testY,pred))


#Visualize Testing Performance
#fig, ax = plt.subplots()
#ax.scatter(pred, testY, edgecolors = (0,0,1))
#ax.plot([testY.min(), testY.max()], [testY.min(), testY.max()],'r--', lw = 3)
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values')
#plt.show(block = True)

