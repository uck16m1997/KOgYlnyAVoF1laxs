# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:54:29 2020

@author: USER
"""

import pandas as pd
import numpy  as np

df = pd.read_csv("term-deposit-marketing-2020.csv")

dropped_df = df.drop(["day","month"],axis=1)

#Create Dummy Variables From  Categorical Variables
cats = ["job","marital","education","default","housing","loan","contact"]
dropped_df.y = dropped_df.y.map(dict(yes=1,no=0))


for c in cats:
    dropped_df[c]=dropped_df[c].astype('category')
dropped_df = pd.get_dummies(dropped_df, drop_first=True)#Drop First Prevents Dummy Variable Trap
y=dropped_df['y']
X=dropped_df.drop(columns=['y'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.linear_model import LogisticRegression #df = 68.99 #dropped_df= 93.01
clf = LogisticRegression(max_iter=5000)

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
scores = {'accuracy' : make_scorer(accuracy_score)}
from sklearn.model_selection import cross_validate
results = cross_validate(estimator = clf, X = X, y = y, cv = 5,scoring=scores)

print("Accuracy: {:.2f} %".format(results["test_accuracy"].mean()*100))


