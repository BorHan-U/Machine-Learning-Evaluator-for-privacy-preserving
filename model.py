# Importing the libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss

from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Import train_test_split function
from sklearn.model_selection import train_test_split


# flask demo ori model........
dataset = pd.read_csv('pima diabetes(email).csv')

# flask demo ori model........
# dataset = pd.read_csv('hiring.csv')

# split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI',
                'Age', 'Glucose', 'BloodPressure', 'Email', 'DiabetesPedigreeFunction']
X = dataset[feature_cols]  # Features
y = dataset.Outcome  # Target variable

# As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
encoder = OrdinalEncoder()
encoder.fit(X)
X_encoded = encoder.transform(X)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

# print(accuracy)
accuracy = round(accuracy_score(y_pred, y_test), 4)

# print('Recall: %.3f' % recall_score(y_test, y_pred))
recall = round(recall_score(y_test, y_pred), 4)
# print(('precision_score(y_test, y_pred)))
precision = round(precision_score(y_test, y_pred), 4)
# print(f1_score(y_test, y_pred))
f1_Score = round(f1_score(y_test, y_pred), 4)
# brier_score
# predict probabilities
probs = clf.predict_proba(X_test)
probs = probs[:, 1]
loss = round(brier_score_loss(y_test, probs), 4)

# Saving model to disk
pickle.dump(accuracy, open('modela.pkl', 'wb'))
modelAccuracy = pickle.load(open('modela.pkl', 'rb'))

# Loading model to compare the results
pickle.dump(recall, open('modelr.pkl', 'wb'))
modelRecall = pickle.load(open('modelr.pkl', 'rb'))

pickle.dump(precision, open('modelp.pkl', 'wb'))
modelPrecision = pickle.load(open('modelp.pkl', 'rb'))

pickle.dump(f1_Score, open('modelf.pkl', 'wb'))
modelF1_Score = pickle.load(open('modelf.pkl', 'rb'))

pickle.dump(loss, open('modell.pkl', 'wb'))
modelLoss = pickle.load(open('modell.pkl', 'rb'))
# predict = model.predict(X_test)
# print(model)


# accuracy = accuracy_score(predict, y_test)
# print(accuracy)
