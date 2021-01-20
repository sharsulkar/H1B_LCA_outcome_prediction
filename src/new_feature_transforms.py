import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def main():
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
            self.input_columns=input_columns

        def date_diff(self,date1,date2):
            """
            Returns the difference between two input dates as timedelta.

            Args:
                date1 (datetime): A date
                date2 (datetime): Another date

            Returns:
                date_difference (timedelta): difference between date1 and date2
            """
            date_difference=date1-date2
            return date_difference

        def is_usa(self,country):
            """
            Checks whether country is 'UNITED STATES OF AMERICA' or not and returns a binary flag

            Args:
                country (str): country

            Returns:
                USA_YN (str): binary flag based on country value
            """
            if country=='UNITED STATES OF AMERICA':
                USA_YN='Y' 
            else:
                USA_YN='N'
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
            X['PROCESSING_DAYS']=self.date_diff(X.DECISION_DATE, X.RECEIVED_DATE).dt.days
            X['VALIDITY_DAYS']=self.date_diff(X.END_DATE, X.BEGIN_DATE).dt.days

            # SOC_Codes
            X['SOC_CD2']=X.SOC_CODE.str.split(pat='-',n=1,expand=True)[0]
            X['SOC_CD4']=X.SOC_CODE.str.split(pat='-',n=1,expand=True)[1].str.split(pat='.',n=1,expand=True)[0]
            X['SOC_CD_ONET']=X.SOC_CODE.str.split(pat='-',n=1,expand=True)[1].str.split(pat='.',n=1,expand=True)[1]

            # USA_YN
            X['USA_YN']=X.EMPLOYER_COUNTRY.apply(self.is_usa)

            # Employer_Worksite_YN
            X['EMPLOYER_WORKSITE_YN']='Y'
            X.loc[X.EMPLOYER_POSTAL_CODE.ne(X.WORKSITE_POSTAL_CODE),'EMPLOYER_WORKSITE_YN']='N'

            # OES_YN
            X['OES_YN']='Y'
            X.iloc[X[~X.PW_OTHER_SOURCE.isna()].index,X.columns.get_loc('OES_YN')]='N'

            # SURVEY_YEAR
            X['SURVEY_YEAR']=pd.to_datetime(X.PW_OES_YEAR.str.split(pat='-',n=1,expand=True)[0]).dt.to_period('Y')
            pw_other_year=X[X.OES_YN=='N'].PW_OTHER_YEAR
            #Rename the series and update dataframe with series object
            pw_other_year.rename("SURVEY_YEAR",inplace=True)
            X.update(pw_other_year)

            # WAGE_ABOVE_PREVAILING_HR
            X['WAGE_PER_HR']=X.WAGE_RATE_OF_PAY_FROM
            #compute for Year
            X.iloc[X[X.WAGE_UNIT_OF_PAY=='Year'].index,X.columns.get_loc('WAGE_PER_HR')]=X[X.WAGE_UNIT_OF_PAY=='Year'].WAGE_RATE_OF_PAY_FROM/2067
            #compute for Month
            X.iloc[X[X.WAGE_UNIT_OF_PAY=='Month'].index,X.columns.get_loc('WAGE_PER_HR')]=X[X.WAGE_UNIT_OF_PAY=='Month'].WAGE_RATE_OF_PAY_FROM/172

            #initialize with WAGE_RATE_OF_PAY_FROM
            X['PW_WAGE_PER_HR']=X.PREVAILING_WAGE
            #compute for Year
            X.iloc[X[X.PW_UNIT_OF_PAY=='Year'].index,X.columns.get_loc('PW_WAGE_PER_HR')]=X[X.PW_UNIT_OF_PAY=='Year'].PREVAILING_WAGE/2067
            #compute for Month
            X.iloc[X[X.PW_UNIT_OF_PAY=='Month'].index,X.columns.get_loc('PW_WAGE_PER_HR')]=X[X.PW_UNIT_OF_PAY=='Month'].PREVAILING_WAGE/172

            X['WAGE_ABOVE_PW_HR']=X.WAGE_PER_HR-X.PW_WAGE_PER_HR

            return X
if __name__ == '__main__':
    main()