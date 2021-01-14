#import libraries
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from pickle import dump, load

#Define transformers that will be used in pipeline
#drop rows
class DropRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, row_index, inplace, reset_index):
      self.row_index = row_index # row index to drop
      self.inplace=True
      self.reset_index=True

    def fit( self, X, y=None):
      return self 
    
    def transform(self, X, y=None):
      X.drop(index=self.row_index,inplace=self.inplace)
      if self.reset_index:
        X.reset_index(inplace=True)
      return X

#build features
class BuildFeaturesTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, input_columns):
    self.input_columns=input_columns

  def date_diff(self,date1,date2):
    return date1-date2

  def is_usa(self,country):
    if country=='UNITED STATES OF AMERICA':
      USA_YN='Y' 
    else:
      USA_YN='N'
    return USA_YN

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
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

##Custom transformer to drop features for input feature list
class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, inplace):
      self.columns = columns # list of categorical columns in input Dataframe
      self.inplace=True

    def fit( self, X, y=None):
      return self 
    
    def transform(self, X, y=None):
      X.drop(columns=self.columns,inplace=self.inplace)
      return X

#Custom transformer to compute Random Standard encoding
#add option to return ordered encoding, whether to include encoding for missing value or not
class RandomStandardEncoderTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, cat_cols, categories=None, RSE=None ):
        self.cat_cols = cat_cols # list of categorical columns in input Dataframe
        self.categories = categories # Array of unique non-numeric values in each categorical column
        self.RSE = RSE # Array of Random Standard encoding for each row in categories
        
    #Return self, nothing else to do here
    def fit( self, X, y=None ):
      #Get a list of all unique categorical values for each column
      self.categories = [X[column].unique() for column in X[self.cat_cols]]
      #replace missing values and append missing value label to each column to handle missing values in test dataset that might not be empty in train dataset
      for i in range(len(self.categories)):
        if np.array(self.categories[i].astype(str)!=str(np.nan)).all():
          self.categories[i]=np.append(self.categories[i],np.nan)
      #compute RandomStandardEncoding 
      self.RSE=[np.random.normal(0,1,len(self.categories[i])) for i in range(len(self.cat_cols))]
      return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y=None):
      for i in range(len(self.cat_cols)):
        #X[str(self.cat_cols[i])].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
        X.loc[:,(str(self.cat_cols[i]))].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
      return X    


#import observations_df for referencing features and corresponding preprocessing actions to be performed on them
observations_df=pd.read_csv('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/reports/final_observations.csv',sep='$',index_col=0,error_bad_lines=False)

#source data 
required_features=list(observations_df[(observations_df.preprocess_comment.isin([np.NaN,'Feature engineering','Target feature','Use feature as is'])) & (~observations_df.preprocess_action.isin(['New feature']))].index)
input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx',usecols=required_features)

#identify and define column sets for applying preprocessing transforms
num_cols=observations_df[(observations_df['Categorical class']=='Numerical') & (observations_df.preprocess_action!='Drop column')].index.values
cat_cols=observations_df[(observations_df['Categorical class'].isin(['Categorical','Ordinal','Binary'])) & (observations_df.preprocess_action!='Drop column') & (observations_df.preprocess_comment!='Target feature')].index.values
drop_cols=set(input_df.columns.values)-set(observations_df[observations_df.preprocess_action.isin(['New feature','Use feature as is'])].index.values)
fe_cols=list(observations_df[(observations_df.preprocess_comment.isin(['Feature engineering'])) & (~observations_df.preprocess_action.isin(['New feature']))].index)

drop_row_index=input_df[~input_df.CASE_STATUS.isin(['Certified','Denied'])].index

#Build preprocessing pipeline
build_feature_pipe=make_pipeline(
    DropRowsTransformer(row_index=drop_row_index,inplace=True,reset_index=True),
    BuildFeaturesTransformer(fe_cols)
    )

numerical_preprocess=make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)
preprocess_pipe=make_column_transformer(
    (DropFeaturesTransformer(columns=list(drop_cols),inplace=True),list(drop_cols)),
    (RandomStandardEncoderTransformer(cat_cols),cat_cols),
    (numerical_preprocess,num_cols),
    remainder='passthrough'
)
all_preprocess=make_pipeline(
    preprocess_pipe
)

#apply pipeline
#feature engineering + drop rows
fe_df=build_feature_pipe.fit_transform(input_df)
#Separate target column - add conditions to apply only on training dataset
y=fe_df.pop('CASE_STATUS')
#drop columns + encoding
X=all_preprocess.fit_transform(fe_df)

#save transformed dataset and target
pd.DataFrame(X,columns=fe_df.columns.values).to_csv('/content/drive/MyDrive/Datasets/processed.csv')

#save pipeline
#reference - https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
dump(all_preprocess,open('/content/drive/MyDrive/preprocess_pipe.pkl','wb'))
#all_preprocess=load(open('/content/drive/MyDrive/preprocess_pipe.pkl','rb'))