# Stone Barnard
# Data Science Final Project
# 21 August 2024

#import pyreader as pyr
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

#did it work?
#carAd.info()

#last check for anything odd
#print(carAd.loc[carAd['Price'].idxmax()])

#initial data analysis of each feature, print a histogram
#for columns in carAd.columns: 
#   plt.figure(figsize=(4,3))
#   sns.histplot(data=carAd,x=columns
#                , bins= 10)
#   plt.title(columns)
#   plt.show()






