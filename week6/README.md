# 🏦 Bank Institution Term Deposit Predictive Model
This project focuses on building a machine learning model to predict whether a customer will subscribe to a term deposit product based on a real-world bank marketing dataset. The solution demonstrates a full data science workflow from data exploration to model prediction, with a focus on handling imbalanced data and selecting the best-performing model.

## 📌 Project Objective
To predict customer subscription to bank term deposits using supervised machine learning techniques, with the goal of supporting marketing decision-making and campaign targeting.

## 🗃️ Dataset
* Source: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* File Used: `bank-additional-full.csv`
* Target Variable: `y` – Whether the customer subscribed to a term deposit (`yes`/`no`)

## Contents
* Notebook: Jupyter notebook file for;
  * Exploratory Data Analysis
  * Developing the code
  
  
* Util.py: python script to import a python library


* Preprocess.py: python script for data preprocessing


* Data.py: python script for data loader


* Model.py: python script for building the model


* Main.py: python script to execute the code using dataset without outliers


* Main_org.py: python script to execute the code using the original dataset


* Requirements.txt: Text file containing python packages needed for the proect.

## Code Execution Guide
* **Notebook File**
  * Exploratory Data Analysis: Run this code first, to get an understanding of the dataset and how the features contribute to the machine learning model.
  * Model: Run this code if you want to generate the python script or you can skip to running the python scripts directly.
  
  
* **Script File**
  * Util.py: Run this code first.
  
  
  * Preprocess.py: This contains the data preprocessing and it should be executed next. It includes functions for encoding categorical columns and rescaling numerical columns.
  
  
  * Data.py: This is the data loader and should be executed after the preprocess file. It contains functions for choosing our dependent and independent variables, splitting the dataset, dimensionality reduction, and treating class imbalance.
  
  
  * Model.py: This contains functions for building the model and making predictions and it is executed after Data.py. The four models considered in this projects are;
    * Logistic Regression
    * XGBoost
    * Multi Layer Perceptron
    * SVM
    
    
  * Main.py/Main_org.py: This is the main code that calls all the methods in the previous scripts and when executed, generates result. This is executed last.
  
 **NOTE:** A quick reminder; Util.py --> Preprocess.py --> Data.py --> Model.py --> Main.py/Main_org.py
