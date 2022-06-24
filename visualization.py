#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 19:07:45 2022

@author: danielzhu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

data_train = pd.read_csv('train.csv')
header = data_train.columns.values
data = data_train.values

def gender():
    female = data_train.loc[data_train.Sex == 'female']['Survived']
    f_survive_rate = sum(female)/len(female)
    print('Female Survive Rate:',f_survive_rate)

    male = data_train.loc[data_train.Sex == 'male']['Survived']
    m_survive_rate = sum(male)/len(male)
    print('Male Survive Rate:',m_survive_rate)
    
    plt.title("Female survive rate {}".format(round(f_survive_rate,2)), size=16)
    female.value_counts().plot.pie(colors=["Blue", "Red"],labels = ['Survivors','Victims'], shadow=True)
    plt.figure()
    plt.title("Male survive rate {}".format(round(m_survive_rate,2)), size=16)
    male.value_counts().plot.pie(colors=["Red", "Blue"],labels = ['Victims','Survivors'], shadow=True)
    return()



def pclass():
    fcls = data_train.loc[data_train.Pclass == 1]['Survived']
    fcls_survive_rate = sum(fcls)/len(fcls)
    print(fcls_survive_rate)
    scls = data_train.loc[data_train.Pclass == 2]['Survived']
    scls_survive_rate = sum(scls)/len(scls)
    print(scls_survive_rate)
    tcls = data_train.loc[data_train.Pclass == 3]['Survived']
    tcls_survive_rate = sum(tcls)/len(tcls)
    print(tcls_survive_rate)
    x = ('First C', 'Second C', 'Thrid C')
    y1 = [len(fcls)-sum(fcls),len(scls)-sum(scls),len(tcls)-sum(tcls)]
    y2 = [sum(fcls),sum(scls),sum(tcls)]
    plt.title('Survive rate of dif class', size=16)
    y = [fcls_survive_rate,scls_survive_rate,tcls_survive_rate]
    bar1 = plt.bar(x, y1, color='r',label='Victims',hatch = '/')
    bar2 = plt.bar(x, y2,bottom=y1, color='g',label = 'Survivors')
    for i in range(3):
        height1 = bar1[i].get_height()+bar2[i].get_height()
        height2 = bar1[i].get_height()
        plt.text(bar1[i].get_x() + bar1[i].get_width() / 2.0, height1, f'{height1-height2:.0f}', ha='center', va='top',size=18)
        plt.text(bar1[i].get_x() + bar1[i].get_width() / 2.0, height2, f'{height2:.0f}', ha='center', va='top',size=18)
        plt.text(bar1[i].get_x() + bar1[i].get_width() / 2.0, height1, '{}%'.format(round(y[i],3)*100), ha='center', va='bottom',size=18)
    plt.legend()
    
def age():
    labels = ["0-7", "7-18", "18-50", "50-90", "UK"]
    data_train["Category"] = data_train["Age"].map(lambda a: " 0-7"   if (0 <= a < 8)  else 
                                                            " 7-18"   if (8 <= a < 19) else 
                                                            " 18-50"  if (19 <= a < 51) else 
                                                            "50-90" if (51 <= a < 90)  else "Unknown")
    age_list = data_train["Category"].drop_duplicates().sort_values().reset_index(drop=True)
    
    age_distribution =  pd.DataFrame({
    "Category": age_list, 
    "Male": age_list.map(lambda age: data_train.loc[(data_train["Category"] == age) & (data_train["Sex"] == "male"), "Category"].count()), 
    "Female": age_list.map(lambda age: data_train.loc[(data_train["Category"] == age) & (data_train["Sex"] == "female"), "Category"].count()), 
    "Total": age_list.map(lambda age: data_train.loc[data_train["Category"] == age, "Category"].count()),
    "MaleSurvivors": age_list.map(lambda age: data_train.loc[(data_train["Category"] == age) & (data_train["Sex"] == "male")]["Survived"].sum()),
    "FemaleSurvivors": age_list.map(lambda age: data_train.loc[(data_train["Category"] == age) & (data_train["Sex"] == "female")]["Survived"].sum()),
    "Survivors": age_list.map(lambda age: data_train.loc[data_train["Category"] == age]["Survived"].sum()) })

    age_distribution["MaleRate"]   = round(100 * age_distribution["MaleSurvivors"] / age_distribution["Male"], 1)
    age_distribution["FemaleRate"] = round(100 * age_distribution["FemaleSurvivors"] / age_distribution["Female"], 1)
    age_distribution["Rate"]       = round(100 * age_distribution["Survivors"] / age_distribution["Total"], 1)
    
    print(age_distribution[["Category", "Male", "Female", "MaleRate", "FemaleRate", "Rate"]].sort_values(by="Category", ascending=True))
    
def fare():
    pass

def embark():
    first = data_train.loc[(data_train.Embarked == 'S')]['Survived']
    third = data_train.loc[(data_train.Embarked == 'C')]['Survived']
    second = data_train.loc[(data_train.Embarked == 'Q')]['Survived']
    
    first_s = sum(first)/len(first)
    second_s = sum(second)/len(second)
    third_s = sum(third)/len(third)
    print(first_s,second_s,third_s)





