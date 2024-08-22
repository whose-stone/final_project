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
#goal is to use model, body type, fuel type, top six used colors. Adding the number of seats, number of doors, and miles

my_df = df[['Genmodel','Color','Bodytype','Fuel_type','Seat_num','Door_num','Price']]
#print(my_df)

#initial data analysis of each feature
#for columns in df.columns: 
#   plt.figure(figsize=(4,3))
#   sns.histplot(data=df,x=columns
#                , bins= 10)
#   plt.title(columns)
#   plt.show()

#remove null values

#see what features have null values
print(my_df.isnull().sum())

#select the models needed: L200, Q3, CX-5, XC90






