
# import all necessary libraries

# import library to get working directory
import os

# import libraies for data manipulation
import pandas as pd
import numpy as np

import Util

# import methods from Preprocess.py
from Preprocess import encode, preprocessed, rescale

# import methods from Data.py
from Data import data_loader, split_data, pca
from Data import dimension_reduction, class_imbalance

# import methods from Model.py
from Model import model, prediction

# Dataset
# function to change working directory
def change_dir(path):
    print("Old directory: ",os.getcwd())
    os.chdir(path)
    print("New directory: ",os.getcwd())
    
# changing the working directory to access the dataset
change_dir('C:\\Users\\PC\\Desktop\\Data Science\\10 Academy\\Training\\Week 6\\Challenge\\Dataset')    

# import the original dataset
dataset = pd.read_csv('bank-additional-full.csv', sep=';')
dataset.name = 'dataset'
print("Original Dataset\n-------------------------")
print(dataset.head())

# changing the working directory to back to original working directory
change_dir('C:\\Users\\PC\\Desktop\\Data Science\\10 Academy\\Training\\Week 6\\Challenge\\Notebooks')

# Preprocessing - Using the new dataset i.e. data without outliers
# replacing basic.4y, basic.6y, basic.9y as basic
dataset['education'] = dataset['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic')

# defining output variable for classification
dataset['subscribed'] = (dataset.y == 'yes').astype('int')

# encoding categorical columns
encoded_data = encode(dataset)
print("Encoded Data\n-------------------------")
print(encoded_data.head())

# preprocessed data
preprocessed_data = preprocessed(dataset)
print("Preprocessed Data\n-------------------------")
print(preprocessed_data.head())

# rescaling numerical columns
preprocessed_data = rescale(preprocessed_data)
print("Rescaled Data\n-------------------------")
print(preprocessed_data.head())

# Data
# dependent and independent variables
X, y = data_loader(preprocessed_data)

# splitting the data
X_train,X_test,y_train,y_test = split_data(X, y)

# pca visualization to get number of components
pca(X_train)

# dimensionality reduction
X_train_reduced, X_test_reduced = dimension_reduction('PCA', 20, X_train, X_test)

# dealing with imbalanced class
X_train_smote, y_train_smote = class_imbalance(X_train_reduced, y_train)

# machine learning model
metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

# 1. Logistic Regression
# KFold cross validation
model_res = model('LR', 'KFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# StratifiedKFold cross validation
model_res = model('LR', 'StratifiedKFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# make prediction
prediction(model_res, 'Linear Regression', X_train_smote, y_train_smote, X_test_reduced, y_test)

# 2. XGBoost
# KFold cross validation
model_res = model('XGB', 'KFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# StratifiedKFold cross validation
model_res = model('XGB', 'StratifiedKFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# make prediction
prediction(model_res, 'XGBoost Classifier', X_train_smote, y_train_smote, X_test_reduced, y_test)

# 3. Multi Layer Perceptron
# KFold cross validation
model_res = model('MLP', 'KFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# StratifiedKFold cross validation
model_res = model('MLP', 'StratifiedKFold', metrics, X_train_smote, X_test_reduced, y_train_smote)
# make prediction
prediction(model_res, 'Multi Layer Perceptron', X_train_smote, y_train_smote, X_test_reduced, y_test)