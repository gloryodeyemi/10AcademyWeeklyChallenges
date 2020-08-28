
# import all necessary libraries

# import methods from data.py
from Data import data_loader, split_data, pca
from Data import dimension_reduction, class_imbalance

# import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# import machine learning model libraries
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

# import evaluation metrics
from sklearn.metrics import accuracy_score,recall_score,precision_recall_curve, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# function to build machine learning models
def model(model, cv_method, metrics, X_train, X_test, y_train):
    if (model == 'LR'):
        # creating an instance of the regression
        model_inst = LogisticRegression()
        print('Logistic Regression\n----------------------')
    elif (model == 'XGB'):
        # creating an instance of the classifier
        model_inst = XGBClassifier()
        print('XGBoost\n----------------------')
    elif (model == 'MLP'):
        # creating an instance of the classifier
        model_inst = MLPClassifier()
        print('Multi Layer Perceptron\n----------------------')
    elif (model == 'SVM'):
        # creating an instance of the classifier
        kernel = input('Enter the kernel (rbf, linear, or poly):')
        model_inst = SVC(kernel=kernel, C=1.0)
        print('Support Vector Classification\n----------------------')
    
    # cross validation
    if (cv_method == 'KFold'):
        print('Cross validation: KFold\n--------------------------')
        cv = KFold(n_splits=10, random_state=100)
    elif (cv_method == 'StratifiedKFold'):
        print('Cross validation: StratifiedKFold\n--------------------------')
        cv = StratifiedKFold(n_splits=10, random_state=100)
    else:
        print('Cross validation method not found!')
    try:
        cv_scores = cross_validate(model_inst, X_train, y_train, 
                                   cv=cv, scoring=metrics)   
        # displaying evaluation metric scores
        cv_metric = cv_scores.keys()
        for metric in cv_metric:
            mean_score = cv_scores[metric].mean()*100
            print(metric+':', '%.2f%%' % mean_score)
            print('')
            
    except:
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        cv_scores = cross_validate(model_inst, X_train, y_train, 
                                   cv=cv, scoring=metrics)
        # displaying evaluation metric scores
        cv_metric = cv_scores.keys()
        for metric in cv_metric:
            mean_score = cv_scores[metric].mean()*100
            print(metric+':', '%.2f%%' % mean_score)
            print('')

    return model_inst
    
# function to make predictions
def prediction(model, model_name, X_train, y_train, X_test, y_test):
    model_ = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #Get the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='.0f')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'conf_{model_name}.png')
    plt.show()