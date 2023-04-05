# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:59:07 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import classification_report
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.head()
df.columns
df = df[['Age', 'Experience', 'Income', 'Family', 'CCAvg','Education', 'Mortgage', 'Securities Account','CD Account', 'Online', 'CreditCard', 'Personal Loan']]
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=69)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train.shape, y_train.shape
import torch
from torch.utils.data import DataLoader, TensorDataset
BATCH_SIZE = 1
train_x = torch.from_numpy(x_train).to(torch.float32)
train_y = torch.from_numpy(y_train).to(torch.float32)
train_x.shape, train_y.shape
(torch.Size([3750, 11]), torch.Size([3750]))
data = TensorDataset(train_x,train_y)
data = DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)
class Model(torch.nn.Module):
     def __init__(self):
        super(Model,self).__init__()
        
        self.layer1 = torch.nn.Linear(11,16)
        self.layer2 = torch.nn.Linear(16,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
     def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
model = Model()
print(model)