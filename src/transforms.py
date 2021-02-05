import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# create logger
module_logger = logging.getLogger('my_application.transforms')

class DropRowsTransformer(BaseEstimator, TransformerMixin):
    """
    A class to drop rows from a DataFrame.

    Args:
        row_index (pandas index object) : A list of indexes that should be dropped from the DataFrame.
        inplace : x (default=True)
        reset_index : binary (default=True)
            Whether reindexing should be performed after drop action
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DropRowsTransformer object.

        Args:
            row_index : pandas index object
                A list of indexes that should be dropped from the DataFrame.
            inplace : binary (default=True)
                Whether the action should be performed inplace or not
            reset_index : binary (default=True)
                Whether reindexing should be performed after drop action
        """
        self.logger = logging.getLogger('my_application.transforms.DropRowsTransformer')
        self.logger.info('creating an instance of DropRowsTransformer')
        self.row_index = None
        self.inplace = True
        self.reset_index = True

    def fit(self, X, y=None):
        """
        Fit the class on input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None
        """
        return self

    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None

        Returns:
            X : Transformed dataframe
        """
        self.row_index=X[~X.CASE_STATUS.isin(['Certified','Denied'])].index
        self.logger.info('Number of rows with CASE_STATUS other than Certified and Denied:%d',self.row_index.shape[0])
        X.drop(index=self.row_index, inplace=self.inplace)
        if self.reset_index:
            X.reset_index(inplace=True,drop=True)
        self.logger.info('Drop rows complete.')
        return X

    def inverse_transform(self,X):
        """
        Inverse transform input.

        Args:
            X (pandas DataFrame): input dataframe
        """
        return self

#Custom transformer to drop features for input feature list
class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
    """A class to drop features from a DataFrame.

    Args:
        columns (array or list): A list of columns that should be dropped from the DataFrame.
        inplace : (binary) : Whether the action should be performed inplace or not, defaulted to True.

    Returns:
        DataFrame : Transformed DataFrame
    """

    def __init__(self, columns, inplace):
        """
        Constructs all the necessary attributes for the DropFeaturesTransformer object.

        Args:
            columns (array or list): A list of columns that should be dropped from the DataFrame.
            inplace : (binary) : Whether the action should be performed inplace or not, defaulted to True.
        """
        self.logger = logging.getLogger('my_application.transforms.DropFeaturesTransformer')
        self.logger.info('creating an instance of DropFeaturesTransformer')
        self.columns = columns  # list of categorical columns in input Dataframe
        self.inplace = True

    def fit(self, X, y=None):
        """
        Fit the class on input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None
        """
        return self

    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None

        Returns:
            X : Transformed dataframe
        """
        #Ensure given columns are present in the DataFrame 
        assert set(self.columns).issubset(set(X.columns.values)),self.logger.error('Columns not found in given input DataFrame.')
        
        X.drop(columns=self.columns, inplace=self.inplace)
        self.logger.info('Drop features complete.')
        return X

    def inverse_transform(self,X):
        """
        Inverse transform input.

        Args:
            X (pandas DataFrame): input dataframe
        """
        return self

class RandomStandardEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    A class to encode categorical features to numeric using random standard normal encoding

    Args:
        cat_cols (array or list): Categorical columns in input Dataframe
        categories (array or list, optional): Array of unique non-numeric values in each categorical column. Defaults to None.
        RSE (array or list): Array of Random Standard encoding for each row in categories. Defaults to None.

    Returns:
        DataFrame : Transformed DataFrame
    """
    #Class Constructor
    def __init__(self, cat_cols, categories=None, RSE=None ):
        """
        Constructs all the necessary attributes for the DropFeaturesTransformer object.

        Args:
            cat_cols (array or list): Categorical columns in input Dataframe
            categories (array or list, optional): Array of unique non-numeric values in each categorical column. Defaults to None.
            RSE (array or list): Array of Random Standard encoding for each row in categories. Defaults to None.
        """
        self.logger = logging.getLogger('my_application.transforms.RandomStandardEncoderTransformer')
        self.logger.info('creating an instance of RandomStandardEncoderTransformer')
        self.cat_cols = cat_cols
        self.categories = None
        self.RSE = None

    def fit(self, X, y=None ):
        """
        Compute the numerical encoding for each categorical column

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None
        """
        #Ensure given columns are present in the DataFrame 
        assert set(self.cat_cols).issubset(set(X.columns.values)),self.logger.error('Columns not found in given input DataFrame.')

        #Get a list of all unique categorical values for each column
        if self.categories is None:
            self.logger.info('Initial encoding computation started.')
            self.categories = [X[column].unique() for column in self.cat_cols]
            #replace missing values and append missing value label to each column to handle missing values in test dataset that might not be empty in train dataset
            for i in range(len(self.categories)):
                if np.array(self.categories[i].astype(str)!=str(np.nan)).all():
                    self.categories[i]=np.append(self.categories[i],np.nan)

            #compute RandomStandardEncoding 
            self.RSE=[np.random.normal(0,1,len(self.categories[i])) for i in range(len(self.cat_cols))]

        else:
            self.logger.info('Updating existing encoding started.')
            for i in range(len(self.cat_cols)):
                #append new unique categories to self.categories
                new_categories=list(set(X[self.cat_cols[i]].unique()).difference(set(self.categories[i])))
                if new_categories!=[]:
                    self.logger.info('Found %d new unique values for %s.',len(new_categories),str(self.cat_cols[i]))
                    
                    self.categories[i]=np.append(self.categories[i],new_categories) #append new categories to the end
                    new_RSE=np.random.normal(0,1,len(new_categories)) #generate new RSE values
                    #regenrate if overlap found with existing encodings
                    if set(new_RSE).issubset(set(self.RSE[i])): 
                        self.logger.info('Found an overlap in existing numerical encoding, trying one more time.')
                        new_RSE=np.random.normal(0,1,len(new_categories))
                    
                    self.RSE[i]=np.append(self.RSE[i],new_RSE) #append new RSE values
                self.logger.info('Updated encodings for %s, new count is %d.',str(self.cat_cols[i]),len(self.categories[i]),)

        self.logger.info('Encoding computation complete.')

        return self

    #Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None

        Returns:
            X : Transformed dataframe
        """
        for i in range(len(self.cat_cols)):
            #replace unseen values with NaN
            X.loc[X[~X[(str(self.cat_cols[i]))].isin(self.categories[i])].index,(str(self.cat_cols[i]))]=np.NaN

            #replace seen values with encoding
            X.loc[:, (str(self.cat_cols[i]))].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
        
        self.logger.info('Encoding application complete.')

        return X

    def inverse_transform(self,X):
        """
        Apply inverse transform to get original values back

        Args:
            X (pandas DataFrame): input dataframe

        Returns:
            X : Dataframe with pre transform inputs
        """
        for i in range(len(self.cat_cols)):
            X.loc[:,(str(self.cat_cols[i]))].replace(dict(zip(self.RSE[i], self.categories[i])),inplace=True)
        return X

#build features
class BuildFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    A class to build new features. 

    Args:
        input_columns (array or list) : The columns that will be used as input for building new features.

    Returns:
        DataFrame : Transformed dataframe with new features added in as columns
    """

    def __init__(self, input_columns):
        """
        Constructs all the necessary attributes for the BuildFeaturesTransformer class instance.

        Args:
            input_columns (array or list) : The columns that will be used as input for building new features.
        """
        self.logger = logging.getLogger('my_application.transforms.BuildFeaturesTransformer')
        self.logger.info('creating an instance of BuildFeaturesTransformer')
        self.input_columns = input_columns

    def date_diff(self, date1, date2):
        """
        Returns the difference between two input dates as timedelta.

        Args:
            date1 (datetime): A date
            date2 (datetime): Another date

        Returns:
            date_difference (timedelta): difference between date1 and date2
        """
        
        date_difference = date1-date2
        self.logger.info('Date difference calculated successfully.')

        return date_difference

    def is_usa(self, country):
        """
        Checks whether country is 'UNITED STATES OF AMERICA' or not and returns a binary flag

        Args:
            country (str): country

        Returns:
            USA_YN (str): binary flag based on country value
        """
        if country == 'UNITED STATES OF AMERICA':
            USA_YN = 'Y'
        else:
            USA_YN = 'N'
        
        #self.logger.info('is_usa function calculated successfully.')

        return USA_YN

    def fit(self, X, y=None):
        """
        Fit the class on input dataframe

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None
        """
        return self

    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe to build new features

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None

        Returns:
            X : Transformed dataframe with new features added in as columns
        """
        #Input columns exist in df
        expected_columns=['DECISION_DATE', 'RECEIVED_DATE','SOC_CODE','EMPLOYER_COUNTRY','EMPLOYER_POSTAL_CODE','WORKSITE_POSTAL_CODE',
        'PW_OTHER_SOURCE','PW_OES_YEAR','PW_OTHER_YEAR','WAGE_RATE_OF_PAY_FROM','WAGE_UNIT_OF_PAY','PW_UNIT_OF_PAY','PREVAILING_WAGE']
        assert set(expected_columns).issubset(set(X.columns.values)),self.logger.error('Columns not found in given input DataFrame.')

        # Processing_Days and Validity_days
        X['PROCESSING_DAYS'] = self.date_diff(X.DECISION_DATE, X.RECEIVED_DATE).dt.days
        X['VALIDITY_DAYS'] = self.date_diff(X.END_DATE, X.BEGIN_DATE).dt.days

        # SOC_Codes
        X['SOC_CD2'] = X.SOC_CODE.str.split(pat='-', n=1, expand=True)[0]
        X['SOC_CD4'] = X.SOC_CODE.str.split(pat='-', n=1, expand=True)[1].str.split(pat='.', n=1, expand=True)[0]
        X['SOC_CD_ONET'] = X.SOC_CODE.str.split(pat='-', n=1, expand=True)[1].str.split(pat='.', n=1, expand=True)[1]

        # USA_YN
        X['USA_YN'] = X.EMPLOYER_COUNTRY.apply(self.is_usa)

        # Employer_Worksite_YN
        X['EMPLOYER_WORKSITE_YN'] = 'Y'
        X.loc[X.EMPLOYER_POSTAL_CODE.ne(X.WORKSITE_POSTAL_CODE), 'EMPLOYER_WORKSITE_YN'] = 'N'

        # OES_YN
        X['OES_YN'] = 'Y'
        X.iloc[X[~X.PW_OTHER_SOURCE.isna()].index,X.columns.get_loc('OES_YN')] = 'N'

        # SURVEY_YEAR
        X['SURVEY_YEAR'] = pd.to_datetime(X.PW_OES_YEAR.str.split(pat='-', n=1, expand=True)[0]).dt.to_period('Y')
        pw_other_year = X[X.OES_YN == 'N'].PW_OTHER_YEAR
        #Rename the series and update dataframe with series object
        pw_other_year.rename("SURVEY_YEAR", inplace=True)
        X.update(pw_other_year)

        # WAGE_ABOVE_PREVAILING_HR
        X['WAGE_PER_HR'] = X.WAGE_RATE_OF_PAY_FROM
        #compute for Year
        X.iloc[X[X.WAGE_UNIT_OF_PAY == 'Year'].index, X.columns.get_loc('WAGE_PER_HR')] = X[X.WAGE_UNIT_OF_PAY == 'Year'].WAGE_RATE_OF_PAY_FROM/2067
        #compute for Month
        X.iloc[X[X.WAGE_UNIT_OF_PAY == 'Month'].index, X.columns.get_loc('WAGE_PER_HR')] = X[X.WAGE_UNIT_OF_PAY == 'Month'].WAGE_RATE_OF_PAY_FROM/172

        #initialize with WAGE_RATE_OF_PAY_FROM
        X['PW_WAGE_PER_HR'] = X.PREVAILING_WAGE
        #compute for Year
        X.iloc[X[X.PW_UNIT_OF_PAY == 'Year'].index, X.columns.get_loc('PW_WAGE_PER_HR')] = X[X.PW_UNIT_OF_PAY == 'Year'].PREVAILING_WAGE/2067
        #compute for Month
        X.iloc[X[X.PW_UNIT_OF_PAY == 'Month'].index, X.columns.get_loc('PW_WAGE_PER_HR')] = X[X.PW_UNIT_OF_PAY == 'Month'].PREVAILING_WAGE/172

        X['WAGE_ABOVE_PW_HR'] = X.WAGE_PER_HR-X.PW_WAGE_PER_HR

        self.logger.info('New features created successfully.')

        return X

    def inverse_transform(self,X):
        """
        Inverse transform input.

        Args:
            X (pandas DataFrame): input dataframe
        """
        return self

#custom transformer for incrementally scaling to standard scale using pooled mean and variance
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    A class to apply standard scaling transform incrementally using pooled mean and variance.

    Args:
        mean (float or array or list): population mean calculated by pooling input sample provided incrementally or in batch mode
        var (float or array or list): population variance calculated by pooling input sample provided incrementally or in batch mode
        n_samples_seen (int or array or list): population size calculated by pooling input sample provided incrementally or in batch mode
        scale (float or array or list): population standard deviation calculated by pooling input sample provided incrementally or in batch mode

    Returns:
        Float or Array : Standard scaled array using population mean and variance
    """

    def __init__(self,mean=None,var=None,n_samples_seen=None,scale=None):
        """
        Constructs all the necessary attributes for the BuildFeaturesTransformer class instance. 

        Args:
            mean (float or array or list): population mean calculated by pooling input sample provided incrementally or in batch mode. Defaults to None.
            var (float or array or list): population variance calculated by pooling input sample provided incrementally or in batch mode. Defaults to None.
            n_samples_seen (int or array or list): population size calculated by pooling input sample provided incrementally or in batch mode. Defaults to None.
            scale (float or array or list): population standard deviation calculated by pooling input sample provided incrementally or in batch mode. Defaults to None.
        """
        self.mean=None 
        self.var=None
        self.n_samples_seen=None
        self.scale=None

    def compute_sample_mean(self,X):
        """
        Compute the mean of input array along the column axis

        Args:
            X (array): input array

        Returns:
            float or array: the computed mean
        """
        return np.mean(X,axis=0)

    def compute_sample_var(self,X):
        """
        Compute the variance of input array along the column axis

        Args:
            X (array): input array

        Returns:
            float or array: the computed variance
        """
        return np.var(X,axis=0)

    def compute_sample_size(self,X):
        """
        Compute the size of input array

        Args:
            X (array): input array

        Returns:
            int: the size of input array
        """
        #assuming X is imputed, if there are null values, throw error aksing that X be imputed first
        return len(X)

    def compute_pooled_mean(self,X):
        """
        Compute the pooled mean using the stored mean values and the input array along the column axis

        Args:
            X (array): input array

        Returns:
            float or array: the computed pooled mean
        """
        #compute the sample mean and size
        sample_mean=self.compute_sample_mean(X)
        sample_count=self.compute_sample_size(X) 
        #compute pool mean
        pool_mean=(self.mean*self.n_samples_seen + sample_mean*sample_count)/(self.n_samples_seen + sample_count)

        return pool_mean

    def compute_pooled_var(self,X):
        """
        Compute the pooled variance using the stored mean values and the input array along the column axis

        Args:
            X (array): input array

        Returns:
            float or array: the computed pooled variance
        """
        #compute the sample var and size
        sample_var=self.compute_sample_var(X)
        sample_count=self.compute_sample_size(X) 
        #compute pool variance
        pool_var=(self.var*(self.n_samples_seen - 1) + sample_var*(sample_count - 1))/(self.n_samples_seen + sample_count - 2)

        return pool_var

    def fit(self,X):
        """
        Fit the class on input array

        Args:
            X (array): input array
        """
        if self.mean is None: 
            self.mean=self.compute_sample_mean(X)
        else: 
            self.mean=self.compute_pooled_mean(X)
        
        if self.var is None:
            self.var=self.compute_sample_var(X)
        else: 
            self.var=self.compute_pooled_var(X)

        if self.n_samples_seen is None:
            self.n_samples_seen=self.compute_sample_size(X) 
        else: 
            self.n_samples_seen+=self.compute_sample_size(X)

        return self

    def transform(self,X):
        """
        Apply the scaling transform on the input array.

        Args:
            X (array): input array

        Returns:
            array: array with standard scaled values
        """
        return (X-self.mean)/np.sqrt(self.var)

    def inverse_transform(self,X):
        """
        Inverse  scaling to return the original values

        Args:
            X (array): scaled input array

        Returns:
            float or array: array with values scaled back to its original scale
        """
        return X*np.sqrt(self.var) + self.mean

