import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os

print("Running submission2_fixed.py...")

# computers = pd.read_csv("../datasets/computers.csv")
# Use absolute path to be safe or relative if running from correct dir
computers = pd.read_csv("dataset/computers.csv")
computers.info()
print(computers.head())
print(computers.tail())
print(computers.isna().sum())

mean_values = computers['Minutes'].mean()
print("Mean Values:", mean_values)

# plt.scatter(computers['Units'],computers['Minutes'],color='purple')
# plt.xlabel('Units')
# plt.ylabel('Minutes')

model0=computers['Minutes'].mean()
model1 = 10+12*computers['Units']
model2 = 6+18*computers['Units']

computers['Model_0']=model0
computers['Model_1']=model1
computers['Model_2']=model2

computers.info()
print(computers)

X = computers[['Units']]
Y = computers['Minutes']
from sklearn.linear_model import LinearRegression 
model3 = LinearRegression()
model3.fit(X,Y)
print("Intercept:",model3.intercept_)
print("Coefficients:",model3.coef_)

# FIX 1: Computers -> computers
model_3 = 4.16 + 15.50*computers['Units']
computers['model_3'] = model_3 

# FIX 2: y -> Y
rsq = model3.score(computers[['Units']],Y)*100
print("RSQ:", rsq)

student = pd.read_csv("dataset/std_marks_data.csv")
print(student)
student.info()
print(student.isna().sum())
student.hours=student.hours.fillna(student.hours.mean())
X = student.iloc[:, :-1]
X.info()
print(X)
Y = student.iloc[:, -1]
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)  
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
print("Model trained successfully.")
