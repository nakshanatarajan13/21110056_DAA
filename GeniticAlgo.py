# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:50:34 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.head()
df=df[['Age', 'Experience', 'Income','Family','CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard','Personal Loan']]
x= df.iloc[:,:-1].values
y= df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=6,activation="relu"))
model.add(Dense(units=6,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics="accuracy")
def fitness(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitnessVal
def roulette_wheel_selection(p):

    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
   
    return ind[0][0]
import copy
def crossover(p1, p2):
    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)
    alpha = np.random.uniform(0, 1, *(c1['position'].shape))
    c1['position'] = alpha*p1['position'] + (1-alpha)*p2['position']
    c2['position'] = alpha*p2['position'] + (1-alpha)*p1['position']

    return c1, c2
def mutate(c, mu, sigma):
    y = copy.deepcopy(c)
    flag = np.random.rand(*(c['position'].shape)) <= mu 
    ind = np.argwhere(flag)
    y['position'][ind] += sigma * np.random.randn(*ind.shape)
  
    return y

def bounds(c, varmin, varmax):

    c['position'] = np.maximum(c['position'], varmin)
    c['position'] = np.minimum(c['position'], varmax)
def ga(costfunc, num_var, varmin, varmax, maxit, npop, num_children, mu, sigma, beta):
  
 
    population = {}
    for i in range(npop):                                                         
        population[i] = {'position': None, 'cost': None}                           
 
    bestsol = copy.deepcopy(population)
    bestsol_cost = np.inf                                                         

  
    for i in range(npop):
        population[i]['position'] = np.random.uniform(varmin, varmax, num_var)    
        population[i]['cost'] = costfunc(population[i]['position'])

        if population[i]['cost'] < bestsol_cost:                                  
            bestsol = copy.deepcopy(population[i])                                  

  
    bestcost = np.empty(maxit)

    for it in range(maxit):

    
        costs = []
        for i in range(len(population)):
            costs.append(population[i]['cost'])                                       
            costs = np.array(costs)
            avg_cost = np.mean(costs)                                                   
            if avg_cost != 0:
                costs = costs/avg_cost
                probs = np.exp(-beta*costs)                                                 
 
    for _ in range(num_children//2):                                             
                                                                               


      
        p1 = population[roulette_wheel_selection(probs)]
        p2 = population[roulette_wheel_selection(probs)]  
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1, mu, sigma)
        c2 = mutate(c2, mu, sigma)
        bounds(c1, varmin, varmax)
        bounds(c2, varmin, varmax)
      
      
        c1['cost'] = costfunc(c1['position'])                                     
      
        if type(bestsol_cost) == float:
             if c1['cost'] < bestsol_cost:                                           
                bestsol_cost = copy.deepcopy(c1)
        else:
            if c1['cost'] < bestsol_cost['cost']:                                  
                bestsol_cost = copy.deepcopy(c1)

      
      
        if c2['cost'] < bestsol_cost['cost']:                                     
            bestsol_cost = copy.deepcopy(c2)

   
    bestcost[it] = bestsol_cost['cost']

   
    

    out = population
    Bestsol = bestsol
    bestcost = bestcost
    return (out, Bestsol, bestcost)
model.fit(x_train, y_train, batch_size = 30, epochs = 100)