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
df.info()

#reset the index
df.reset_index(drop=True, inplace=True)

#select those features that lend themselves to rendom forest analysis
#goal is to use model, body type, fuel type, top six used colors, the number of seats, the number of doors.

carAd = df[['Genmodel','Color','Bodytype','Fuel_type','Seat_num','Door_num','Price']]
#print(carAd)

#Start cleaning the data

#see what the car model names are
print(carAd["Genmodel"].value_counts(dropna = False)) 

name_list = [name for name in carAd['Genmodel']]

#list to select specific models
models = ['L200', 'Q3', 'CX-5', 'XC90']

#are they in our list
#for model in models:
#    if model in name_list:
#        print(f"{model} is in the list.")
#    else:
#        print(f"{model} is not in the list")

#select our models from the Genmodel column
carAd_models = carAd[carAd['Genmodel'].isin(models)]
print(len(carAd_models))



#initial data analysis of each feature
#for columns in df.columns: 
#   plt.figure(figsize=(4,3))
#   sns.histplot(data=df,x=columns
#                , bins= 10)
#   plt.title(columns)
#   plt.show()






