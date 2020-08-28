
# import library for preprocessing
from sklearn.preprocessing import StandardScaler

# import libraies for data manipulation
import pandas as pd

import Util

# preprocessing class
class preprocess:
    
    # create list containing categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    # create list containing numerical columns
    num_cols = ['duration', 'campaign', 'emp.var.rate',"pdays","age", 'cons.price.idx', 
                'cons.conf.idx', 'euribor3m', 'nr.employed', 'previous']
    
    # function to encode categorical columns
    def encode(self, data):
        cat_var_enc = pd.get_dummies(data[self.cat_cols], drop_first=False)
        return cat_var_enc
    
    # function to 
    def preprocessed(self, data):
        # adding the encoded columns to the dataframe
        data = pd.concat([data, self.encode(data)], axis=1)
        # saving the column names of categorical variables
        cat_cols_all = list(self.encode(data).columns)
        # creating a new dataframe with features and output
        cols_input = self.num_cols + cat_cols_all
        preprocessed_data = data[cols_input + ['subscribed']]
        return preprocessed_data
    
    # function to rescale numerical columns
    def rescale(self, data):
        # creating an instance of the scaler object
        scaler = StandardScaler()
        data[self.num_cols] = scaler.fit_transform(data[self.num_cols])
        return data
    
# create class methods
preprocess.encode = classmethod(preprocess.encode)
encode = preprocess.encode
preprocess.preprocessed = classmethod(preprocess.preprocessed)
preprocessed = preprocess.preprocessed
preprocess.rescale = classmethod(preprocess.rescale)
rescale = preprocess.rescale