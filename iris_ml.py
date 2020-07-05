# Iris Classification Problem

import pandas as pd
import numpy as np
import seaborn as sns
import requests
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()

print(iris)
# assign data and target to their own variables
x = iris.data
y = iris.target

# since we are training and testing, we need to split our dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5) # splits the data 50/50

# Now we need to choose a classification algorithm, I will go with the KNN.
'''
KNN is based on feature simularity or how closely features resemble our training set.
An object is classified by being assigned to the class most common among 'k' nearest neighbors.
This will work perfectly for the iris dataset.
'''
classifier=neighbors.KNeighborsClassifier()

# Now we have created the model, we can now train it to predict which flower belongs to which species
q = classifier.fit(x_train,y_train)


# We can now predict
pred = classifier.predict(x_test)

print("Model accuracy:",accuracy_score(y_test,pred))
