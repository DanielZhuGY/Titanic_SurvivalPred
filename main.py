#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:03:48 2022

@author: danielzhu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

## Import the training data
data_train = pd.read_csv('train.csv')
data_train = data_train.fillna(0)
header = data_train.columns.values
data_train = data_train.values

y_train = data_train[:,1]
x_train = []
x_need = [2,4,5,9]

for i in x_need:
  x_train.append(data_train[:,i])

_,n = np.shape(x_train)

## Data Pretreatment
## Numerical Expression
for i in range(n):
  if x_train[1][i] == 'male':
    x_train[1][i] = 1
  else:
    x_train[1][i] = 0
  x_train[2][i] = int(x_train[2][i])
  if x_train[3][i]<50:
    x_train[3][i] = 0
  elif x_train[3][i]<150:
    x_train[3][i] = 1
  else:
    x_train[3][i] = 2
    
## Model Building
## LogisticRegression, SGD(Batchsize = 100), iteration = 20000

class LogisticRegression:
  def __init__(self,x,y):
    _,self.n = np.shape(x)

    self.X = x.T ; self.Y = y.reshape(-1)
    print(np.shape(self.X))

  def sigmoid(self,x):
    return(1/(1+np.exp(-x)))
  
  def h(self,B_x,B_y,theta):
    pred = []
    for x in B_x:
      pred.append(self.sigmoid(theta.dot(x)[0]))
    
    return(np.array(pred))

  def loss(self,x,y,theta):

    hf = self.h(x,y,theta)

    error = hf-y
    
    dJ = np.dot(error.reshape(1,self.batch_size),x)/self.batch_size


    return(error,dJ)

  def SGD(self,iters,step,theta,batch_size):
    self.batch_size = batch_size
    error_list = []
    for iter in range(iters):

      if iter%1000==0:
        print('No.',iter)

      x_batch = []
      y_batch = []
      for i in range(batch_size):
        index = random.randint(0,self.n-1)## No repeat might be better
        x_batch.append(self.X[index])
        y_batch.append(self.Y[index])


      error,dJ = self.loss(np.array(x_batch),np.array(y_batch),theta)


      theta=theta-step*dJ
      error_list.append(sum(error**2))
    return(theta,error_list)

  def Regression(self,iter,LR,theta):
    
    theta_h,error = self.SGD(iter,LR,theta,100)
    return(theta_h,error)

  ### Here to process the simulation
x = np.array(np.vstack((x_train,np.ones((n)))))

p = LogisticRegression(x,y_train)

theta = np.zeros((1,5))
theta_h,error = p.Regression(20000,0.01,theta)##(iter,learning rate)

print('Model finish!!!')

## Visualize the fitting process
plt.plot(error)
plt.ylabel('Error')
plt.xlabel('Iteration')

###== This part is for prediction ==###
## Import the test data
data_test = pd.read_csv('test.csv')
data_test = data_test.fillna(0)
t_header = data_test.columns.values
data_test = data_test.values

x_test = []

## Data Pretreatment
x_need_test = [1,3,4,8]
for i in x_need_test:
  x_test.append(data_test[:,i])
print(np.shape(x_test))
_,n_t = np.shape(x_test)

for i in range(n_t):
  if x_test[1][i] == 'male':
    x_test[1][i] = 1
  else:
    x_test[1][i] = 0
  x_test[2][i] = int(x_test[2][i])
  if x_test[3][i]<50:
    x_test[3][i] = 0
  elif x_test[3][i]<150:
    x_test[3][i] = 1
  else:
    x_test[3][i] = 2

x_test = np.vstack((x_test,np.ones((n_t))))
y_test = np.dot(theta_h,x_test)
y_test_dum = y_test.T

## If the probability of living is greater than 50%, we assume this man survive.
y_est = []
for i in range(len(y_test_dum)):
  y_test_dum[i] = (1/(1+np.exp(-int(y_test_dum[i]))))
  if y_test_dum[i]>=0.5:
    y_est.append(1)
  else:
    y_est.append(0)
y_test = y_test_dum.T
result = np.vstack((data_test[:,0],y_est))

## Generate the final result
import csv
with open('eggs.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=",")
  writer.writerow(('PassengerId','Survived'))
  writer.writerows(result.T)