# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:43:41 2023

@author: user
"""

#Load our libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load our data
df = pd.read_csv("C:/Users/user/Desktop/heart_deployment/heart_clean_p.csv")

columns = ['AgeCategory','DiffWalking','Stroke', 'PhysicalHealth', 'Diabetic','HeartDisease']
df = df[columns]

X = df.iloc[:, 0:5]
y = df.iloc[:, 5:]
#splitting our data into training and test
from sklearn.model_selection import train_test_split
np.random.seed(42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
#define a model
rf= RandomForestClassifier()
#fit the model
rf.fit(X_train.values,y_train['HeartDisease'].values)
import pickle
pickle.dump(rf, open('model_hrt.pkl', 'wb'))
