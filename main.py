#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:10:34 2022

@author: danielzhu
"""
import numpy as np
import pandas as pd
import random
import csv

## Import the training data
data_train = pd.read_csv('train.csv')
data_train = data_train.fillna(0)
header = data_train.columns.values
data_train = data_train.values

###== Data Pretreatment ==###
y_train = data_train[:,1]
x_train = []
x_need = [2,4,5,11]

for i in x_need:
  x_train.append(data_train[:,i])


_,n = np.shape(x_train)


## Numerical Expression
for i in range(n):

  if x_train[0][i] >2:
    x_train[0][i] = 1
  else:
    x_train[0][i] = -1

  if x_train[1][i] == 'male':
    x_train[1][i] = -1
  else:
    x_train[1][i] = 1
    

  if x_train[2][i] == 0:
    x_train[2][i] = 1
  elif x_train[2][i]<8:
    x_train[2][i] = 0
  elif x_train[2][i]<90:
    x_train[2][i] = -1

  if x_train[3][i] == 'C':
    x_train[3][i] = 1
  else:
    x_train[3][i] = -1
    
x_train = np.array(x_train).T

###== LogisticRegression ==###
class LogisticRegression:
    def __init__(self,x,y):
        n,d = np.shape(x)
        self.x = np.hstack((x,np.ones((n,1)))) ; self.y = y.reshape(-1)
        
        self.d = d+1 ; self.n = n
        self.theta = np.zeros((self.d,1))

    
    def sigmoid(self,x):
        return(float(1/(1+np.exp(-x.astype(float))))) ###important tips
    
    def J(self,theta,x,y):
        error = [y[i]-self.sigmoid(x[i]@theta) for i in range(self.batch_size)]


        dJ = -x.T@error/self.batch_size

        return(error,dJ)
    
    def loss(self,theta):

        t = [self.y[i]*np.log(self.sigmoid(self.x[i]@theta))+(1-self.y[i])*(np.log(1-self.sigmoid(self.x[i]@theta))) for i in range(self.n)]
        '''
        t = [(self.y[i]-self.sigmoid(self.x[i]@theta))**2]
        '''
        return(-sum(t)/self.n)


    def SGD(self,batch_size,epoches,iters,LR):
        self.batch_size = batch_size
        error_l = []
        theta = self.theta
        for epoch in range(epoches):

            index = random.sample(range(0,self.n),batch_size)
            batch_x = [self.x[i] for i in index]
            batch_y = [self.y[i] for i in index]
            if epoch%10 == 0:
              print('The {} epoch'.format(epoch),'\n Loss:{} Theta:{}'.format(self.loss(theta),theta))
            for ite in range(iters):
              error,dJ = self.J(theta,np.array(batch_x),np.array(batch_y).reshape(batch_size,1))

              theta = theta - LR*dJ
            '''
            theta_optimizer[1] = self.loss(theta)
            if theta_optimizer[1]>theta_optimizer[2]:
              theta = theta_optimizer[0]
            else:
              theta_optimizer[2] = theta_optimizer[1]
            '''
            error_l.append(self.loss(theta))
        print('Finished!!')
        return(theta,error_l)

      ## Run the regression ##
    
regre = LogisticRegression(x_train, y_train)
h_theta,error = regre.SGD(80,2000,30,0.001) ## (Batch_size, Epoch, Iteration, Learning_Rate)

###== Processing Test Data(Same as pretreatment) ==###
data_test = pd.read_csv('test.csv')
data_test = data_test.fillna(0)
t_header = data_test.columns.values
data_test = data_test.values

x_test = []
print(t_header)
x_need_test = [1,3,4,10]
for i in x_need_test:
  x_test.append(data_test[:,i])
_,n_t = np.shape(x_test)

for i in range(n_t):
  if x_test[0][i] > 2:
    x_test[0][i] = 1
  else:
    x_test[0][i] = -1
  if x_test[1][i] == 'male':
    x_test[1][i] = -1
  else:
    x_test[1][i] = 1
  x_test[2][i] = int(x_test[2][i])

  if x_test[2][i] == 0:
    x_test[2][i] = 1
  elif x_test[2][i]<8:
    x_test[2][i] = 0
  elif x_test[2][i]<90:
    x_test[2][i] = -1

  if x_test[3][i] == 'C':
    x_test[3][i] = 1
  else:
    x_test[3][i] = 0

x_test = np.vstack((x_test,np.ones((n_t))))
y_test = np.dot(x_test.T,h_theta)


y_test_dum = y_test
y_est = []
for i in range(len(y_test_dum)):
  y_test_dum[i] = (1/(1+np.exp(-int(y_test_dum[i]))))
  if y_test_dum[i]>0.5:
    y_est.append(1)
  else:
    y_est.append(0)
y_test = y_test_dum.T
result = np.vstack((data_test[:,0],y_est))

### Log the Result ##

with open('result.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=",")
  writer.writerow(('PassengerId','Survived'))
  writer.writerows(result.T)
