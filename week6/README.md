# WEEK 6 CHALLENGE

## Project
Bank Institution Term Deposit Predictive Model

## Business need
You successfully finished up to your rigorous job interview process with Bank of Portugal as a machine learning researcher. The investment and portfolio department would want to be able to identify their customers who potentially would subscribe to their term deposits. As there has been heightened interest of marketing managers to carefully tune their directed campaigns to the rigorous selection of contacts, the goal of your employer is to find a model that can predict which future clients who would subscribe to their term deposit. Having such an effective predictive model can help increase their campaign efficiency as they would be able to identify customers who would subscribe to their term deposit and thereby direct their marketing efforts to them. This would help them better manage their resources (e.g human effort, phone calls, time)


The Bank of Portugal, therefore, collected a huge amount of data that includes customers profiles of those who have to subscribe to term deposits and the ones who did not subscribe to a term deposit. As their newly employed machine learning researcher, they want you to come up with a robust predictive model that would help them identify customers who would or would not subscribe to their term deposit in the future.


Your main goal as a machine learning researcher is to carry out data exploration, data cleaning, feature extraction, and developing robust machine learning algorithms that would aid them in the department.

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
