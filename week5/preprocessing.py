import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(filename='steps.log',level=logging.DEBUG)
class preprocess():

    '''
    A class to handle the preprocessing steps
    methods:
        array2df(data)
            - To turn a list to a series
        extract_features(X, for_predict)
            - To engineer features from any giving dataframe or series
        
    '''

  #Initializing

  def message():
    print("The data is expected to have 7 columns corresponding to the columns\
            in this order ['Store', 'DayofWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday']\n\
                    - Store: int\n\
                    - DayofWeek: int\n\
                    - Date: str ('yyyy-mm-dd')\n\
                    - Open: int\n\
                    - Promo: int\n\
                    - StateHoliday: str ['a','b','c','o']\n\
                    - SchoolHoliday: int")
  def __init__(self, X = None, cols = None, for_predict = True):
    self.X = X
    self.for_predict = for_predict

    if not cols is None:
      self.cols = cols
    else:
      self.cols = ['Store', 'DayofWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday']


  def array2df(self, data,  dim1 = True):

      '''
        Docstring
        Turn a list or array to  a dataframe.

        Parameters
        ----------
        data : list or array
                Should contain the values to be turned to a dataframe
        dim1 :  bool, default = True
                specify the dimension data. data is 1D if true

        Return
        ------
        Dataframe
            Dataframe format of the data
        
        Notes
        ------
        The data is expected to have 7 columns corresponding to the columns
            in this order ['Store', 'DayofWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday']
                    - Store: int
                    - DayofWeek: int
                    - Date: str ('yyyy-mm-dd')
                    - Open: int
                    - Promo: int
                    - StateHoliday: str ['a','b','c','o']
                    - SchoolHoliday: int

        Example
        -------
        test_data = array2df([100, 7, '2013-02-02', 1, 1, 'a', 1])
        test_data
                Store   DayofWeek   Date     Open   Promo   StateHoliday    SchoolHoliday
            0    100        7     2013-02-02   1      1          'a'               1
        
        '''
    #Turn list to dataframe and return dataframe
    if dim1:
      try:
        #data = np.array(data)
        df = pd.DataFrame([data], columns=cols)
        return df
        logging.info('1D value transformed to dataframe')
      except:
        logging.warning("Error from the input")
        message()
    else:
        try:
            df = pd.DataFrame([data], columns=cols)
            message()
        except:
            logging.warning("Error from the input")
            message()

  def extract_features(self, X = None, for_predict = None):

    '''
    Docstring
    This Function is for feature engineering

    PARAMETERS:
    -----------
    X: dataframe
        A dataframe with the same number of columns and column names as the test data
        Expected column names ['Store', 'DayofWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday']

    for_predict : bool
                    To set to predicting mode
    
    return:
    ------
      X:
        A dataframe with engineered features
    
     Notes
        ------
        The data is expected to have 7 columns corresponding to the columns
            in this order ['Store', 'DayofWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday']
                    - Store: int
                    - DayofWeek: int
                    - Date: str ('yyyy-mm-dd')
                    - Open: int
                    - Promo: int
                    - StateHoliday: str ['a','b','c','o']
                    - SchoolHoliday: int
    '''
    #Check if data was passed
    if not X is None:
      self.X = X
    if not for_predict is None:
      self.for_predict = for_predict
    
    if self.for_predict:
      try:
        self.X = array2df(self.X)
        logging.info("Turn values passed to a dataframe")
      except:
        logging.warning('There is a problem in the values passed')
        message

    # if X.columns[0] == 'Id':
    #   X.drop(['Id'], axis = 1)
    try:
      self.X['Date'] = pd.to_datetime(self.X['Date'])
    except:
      logging.warning('Check date format - yyyy-mm-dd')
      message


    #New features
    self.X['year'] = self.X.Date.apply(lambda x: x.year )
    self.X['month'] = self.X.Date.apply(lambda x: x.month)
    self.X['dow'] = self.X.Date.apply(lambda x: x.dayofweek )
    # X['day_name'] = X.Date.apply(lambda x: x.day_name() )
    # X['month_name'] = X.Date.apply(lambda x: x.month_name() )
    self.X['doy'] = self.X.Date.apply(lambda x: x.dayofyear )
    self.X['quarter'] = self.X.Date.apply(lambda x: x.quarter )
    self.X['month_start'] = self.X.Date.apply(lambda x: 1 if x.is_month_start else 0 )
    self.X['month_end'] = self.X.Date.apply(lambda x: 1 if x.is_month_end else 0)
    self.X['is_weekend'] = self.X.dow.apply(lambda x: 1 if x < 5 else 0)
    self.X['dom'] = self.X.Date.apply(lambda x: x.day)
    self.X['evenDays'] = self.X.dom.apply(lambda x: 0 if x%2 else 1)

    self.X.drop(['dom','Date'], axis = 1, inplace = True)

    return self.X
