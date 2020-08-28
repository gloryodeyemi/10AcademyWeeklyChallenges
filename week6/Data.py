
# importing all necessary libraries

# import methods from Preprocess.py
from Preprocess import encode, preprocessed, rescale

# import libraies for data manipulation
import pandas as pd

# import libraries for visualization
import matplotlib.pyplot as plt

# import library for splitting dataset
from sklearn.model_selection import train_test_split

# import library for dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import library for dealing with class imbalance
from imblearn.over_sampling import SMOTE

# function to get the dependent and independent variable
def data_loader(data):
    X = data.drop(columns=[ "subscribed", 'duration'])
    y = data["subscribed"]
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    return X,y

# function to split dataset
def split_data(X, y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=1)
    # printing the shape of training set
    print(f'Train set X shape: {X_train.shape}')
    print(f'Train set y shape: {y_train.shape}')
    # printing the shape of test set
    print(f'Test set X shape: {X_test.shape}')
    print(f'Test set y shape: {y_test.shape}')
    return X_train,X_test,y_train,y_test

# function to get the number of components for dimensionality reduction
def pca(data):
    # create an instance of pca
    pca = PCA()
    # fit pca to our data
    pca.fit(data)
    # saving the explained variance ratio
    explained = pca.explained_variance_ratio_
    # plot the cumulative variance explained by total number of components
    plt.figure(figsize=(12,6))
    plt.plot(range(1,61), explained.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cummulative Explained Variance')
    plt.savefig('pca.png')
    plt.show()

# function to reduce dimensions
def dimension_reduction(method, components, train_data, test_data):
    # PCA
    if (method == 'PCA'):
        pca = PCA(n_components=components)
        pca.fit(train_data)
        pca_train = pca.transform(train_data)
        X_train_reduced = pd.DataFrame(pca_train)
        print("original shape:   ", train_data.shape)
        print("transformed shape:", X_train_reduced.shape)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        # applying method transform to X_test
        pca_test = pca.transform(test_data)
        X_test_reduced = pd.DataFrame(pca_test)
        
    # TSNE
    elif (method == 'TSNE'):
        tsne = TSNE(n_components=components)
        tsne_train = tsne.fit_transform(train_data)
        X_train_reduced = pd.DataFrame(tsne_train)
        print("original shape:   ", train_data.shape)
        print("transformed shape:", X_train_reduced.shape)
        # applying method transform to X_test
        tsne_test = tsne.fit_transform(test_data)
        X_test_reduced = pd.DataFrame(tsne_test)
    
    else:
        print('Dimensionality reduction method not found!')
        
    return X_train_reduced, X_test_reduced

# function to deal with imbalanced class
def class_imbalance(X_data, y_data):
    # creating an instance
    sm = SMOTE(random_state=27)
    # applying it to the data
    X_train_smote, y_train_smote = sm.fit_sample(X_data, y_data)
    return X_train_smote, y_train_smote