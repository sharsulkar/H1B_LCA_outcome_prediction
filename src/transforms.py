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

    def __init__(self, row_index, inplace, reset_index):
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
        self.row_index = row_index
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
        X.drop(index=self.row_index, inplace=self.inplace)
        if self.reset_index:
            X.reset_index(inplace=True)
        self.logger.info('Drop rows complete.')
        return X

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
        X.drop(columns=self.columns, inplace=self.inplace)
        self.logger.info('Drop features complete.')
        return X

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
        self.categories = categories
        self.RSE = RSE

    def fit(self, X, y=None ):
        """
        Compute the numerical encoding for each categorical column

        Args:
            X (pandas DataFrame): input dataframe
            y : place holder, defaulted to None
        """
          #Get a list of all unique categorical values for each column
        self.categories = [X[column].unique() for column in X[self.cat_cols]]
        #replace missing values and append missing value label to each column to handle missing values in test dataset that might not be empty in train dataset
        for i in range(len(self.categories)):
            if np.array(self.categories[i].astype(str) != str(np.nan)).all():
                self.categories[i] = np.append(self.categories[i], np.nan)
        #compute RandomStandardEncoding
        self.RSE = [np.random.normal(0, 1, len(self.categories[i])) for i in range(len(self.cat_cols))]

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
            X.loc[:, (str(self.cat_cols[i]))].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
        
        self.logger.info('Encoding application complete.')

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
        Constructs all the necessary attributes for the BuildFeaturesTransformer object.

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
        
        self.logger.info('is_usa function calculated successfully.')

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
