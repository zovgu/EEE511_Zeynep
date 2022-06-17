# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:52:30 2022

@author: Zeyneep Övgü Yaycı
"""


import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# read the dataset with the attributes as the title Optimal depth of win = ODOW
dataset = pd.read_csv("krkopt.data", names = ["White King File","White King Rank", "White Rook File", "White Rook Rank", "Black King File", "Black King Rank", "ODOW" ])
data = dataset.values

#separate input and output
X = data[:,:-1].astype(str)
y = data[:, -1].astype(str)


# encode files 
o_enc = OrdinalEncoder()
X_enc = o_enc.fit_transform(X)
l_enc = LabelEncoder()
y_enc = l_enc.fit_transform(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_enc,y_enc, test_size=0.3, shuffle=True, random_state=1)

# MLP
print("MLP with sklearn")
mlp = MLPClassifier(hidden_layer_sizes=(250,200,150,100,50), max_iter=1000, activation='relu', solver='adam', random_state=1)
mlp.fit(Xtrain,Ytrain)
YpredMLP = mlp.predict(Xtest)

print("Classification Report MLP")
crMLP = classification_report(Ytest,YpredMLP)
print(crMLP)

print("Confusion Matrix MLP")
ConfMatMLP = confusion_matrix(Ytest, YpredMLP)
print(ConfMatMLP)

CrossValMLP = KFold(n_splits=10, random_state=1, shuffle=True)
CVScoresMLP = cross_val_score(mlp, Xtest, Ytest, scoring='accuracy', cv = CrossValMLP, n_jobs=-1)
print("Cross Validation MLP")
print(CVScoresMLP)
print("Training set loss: %f" % mlp.loss_)



