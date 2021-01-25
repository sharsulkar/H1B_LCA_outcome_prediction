import numpy as np
import pandas as pd
import logging
from mylib import read_csv_to_list, modify_observations, missing_statistics, cardinality_statistics
from transforms import BuildFeaturesTransformer, DropRowsTransformer

#load data
required_features=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/required_features.csv',header=None,squeeze=True)
try :
    input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx',usecols=required_features)
except ValueError:
    logging.exception('ValueError: columns %s not found',required_features)

#build features
fe_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
drop_row_index=input_df[~input_df.CASE_STATUS.isin(['Certified','Denied'])].index

#Build preprocessing pipeline
build_feature_pipe=make_pipeline(
    transforms.DropRowsTransformer(row_index=drop_row_index,inplace=True,reset_index=True),
    transforms.BuildFeaturesTransformer(fe_cols)
)

#apply preprocessing pipeline
transformed_df=build_feature_pipe.transform(input_df)

#Recreate observations file
#store feature statictics in a dataframe
df_data_statistics=pd.DataFrame(data=None,
                                index=data.columns,
                                columns=['Dtype','percent_missing','cardinality','preprocess_action','preprocess_comment','new_feature_name','new_feature_logic','categorical_class','embedding']
)


#dataframe statistics
for column in data.columns:
  #identify numeric, non-numeric and date columns
  df_data_statistics.Dtype.loc[column]=data[column].dtype
  #% missing data for each column
  df_data_statistics.percent_missing.loc[column]=(data.shape[0]-data[column].count())*100/data.shape[0]
  #Cardinality of each column
  df_data_statistics.cardinality.loc[column]=(data.shape[0]-len(data[column].unique()))*100/data.shape[0]

#drop features with missing values >threshold
missing_threshold=40.0
for idx in df_data_statistics[df_data_statistics.percent_missing>=missing_threshold].index:
  df_data_statistics.loc[[idx],['preprocess_action','preprocess_comment']]=['Drop column','missing values>='+str(missing_threshold)+'% of total']

#drop features with cardinality>threshold
cardinality_threshold=80.0
for idx in df_data_statistics[df_data_statistics.cardinality<80.0].index:
  df_data_statistics.loc[[idx],['preprocess_action','preprocess_comment']]=['Drop column','High Cardinality, threshold '+str(cardinality_threshold)+'% of total']

#Separate target column
df_data_statistics.loc[['CASE_STATUS'],['preprocess_action','preprocess_comment']]=['Pop column into a separate list','Target feature']

#FEATURE Engineering - date columns
#Create a new feature - PROCESSING_DAYS from 'RECEIVED_DATE', 'DECISION_DATE'
df_data_statistics.loc[['RECEIVED_DATE', 'DECISION_DATE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','PROCESSING_DAYS','days(DECISION_DATE-RECEIVED_DATE)']
#Create a new feature - VALIDITY_DAYS from 'BEGIN_DATE', 'END_DATE'
df_data_statistics.loc[['BEGIN_DATE', 'END_DATE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','VALIDITY_DAYS','days(END_DATE-BEGIN_DATE)']

#Feature engineering - split SOC_CODE into 2 new features - SOC_CODE_2, SOC_CODE_4
df_data_statistics.loc[['SOC_CODE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','SOC_CODE_2,SOC_CODE_4','SOC_CODE.split(\'-\')']

#Feature engineering - EMPLOYER_COUNTRY - US or NOT
df_data_statistics.loc[['EMPLOYER_COUNTRY'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','USA_YN','IF EMPLOYER_COUNTRY==USA THEN Y ELSE N END']

#Drop columns - EMPLOYER_* except 'EMPLOYER_NAME',EMPLOYER_POSTAL_CODE
emp_cols=['TRADE_NAME_DBA','EMPLOYER_ADDRESS1','EMPLOYER_ADDRESS2','EMPLOYER_CITY','EMPLOYER_STATE',
          'EMPLOYER_COUNTRY','EMPLOYER_PROVINCE','EMPLOYER_PHONE','EMPLOYER_PHONE_EXT','EMPLOYER_POC_LAST_NAME',
          'EMPLOYER_POC_FIRST_NAME','EMPLOYER_POC_MIDDLE_NAME','EMPLOYER_POC_JOB_TITLE','EMPLOYER_POC_ADDRESS1',
          'EMPLOYER_POC_ADDRESS2','EMPLOYER_POC_CITY','EMPLOYER_POC_STATE','EMPLOYER_POC_POSTAL_CODE',
          'EMPLOYER_POC_COUNTRY','EMPLOYER_POC_PROVINCE','EMPLOYER_POC_PHONE','EMPLOYER_POC_PHONE_EXT','EMPLOYER_POC_EMAIL']
df_data_statistics.loc[emp_cols,['preprocess_action','preprocess_comment']]=['Drop column','Not Useful']

#Drop columns - AGENT_* AGENT_REPRESENTING_EMPLOYER
agt_cols=['AGENT_ATTORNEY_LAST_NAME','AGENT_ATTORNEY_FIRST_NAME','AGENT_ATTORNEY_MIDDLE_NAME','AGENT_ATTORNEY_ADDRESS1',
          'AGENT_ATTORNEY_ADDRESS2','AGENT_ATTORNEY_CITY','AGENT_ATTORNEY_STATE','AGENT_ATTORNEY_POSTAL_CODE',
          'AGENT_ATTORNEY_COUNTRY','AGENT_ATTORNEY_PROVINCE','AGENT_ATTORNEY_PHONE','AGENT_ATTORNEY_PHONE_EXT',
          'AGENT_ATTORNEY_EMAIL_ADDRESS','LAWFIRM_NAME_BUSINESS_NAME','STATE_OF_HIGHEST_COURT','NAME_OF_HIGHEST_STATE_COURT'      
]
df_data_statistics.loc[agt_cols,['preprocess_action','preprocess_comment']]=['Drop column','Not Useful']

#Drop columns -SECONDARY_ENTITY_BUSINESS_NAME
df_data_statistics.loc['SECONDARY_ENTITY_BUSINESS_NAME',['preprocess_action','preprocess_comment']]=['Drop column','Not Useful']

#Drop columns - WORKSITE_* except WORKSITE_POSTAL_CODE
wkst_cols=['WORKSITE_ADDRESS1','WORKSITE_ADDRESS2','WORKSITE_CITY','WORKSITE_COUNTY','WORKSITE_STATE']
df_data_statistics.loc[wkst_cols,['preprocess_action','preprocess_comment']]=['Drop column','Not Useful']
#Feature engineering - Worksite same as employer address 
df_data_statistics.loc[['WORKSITE_POSTAL_CODE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','EMPLOYER_WORKSITE_YN','IF EMPLOYER_POSTAL_CODE==WORKSITE_POSTAL_CODE THEN Y ELSE N END']

#Feature engineering - convert PREVAILING_WAGE and WAGE_RATE_OF_PAY_FROM to hourly wage - if PW_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067
#Feature engineering - WAGE_ABOVE_PREVAILING_HR = WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR
df_data_statistics.loc[['PREVAILING_WAGE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','PREVAILING_WAGE_HR;WAGE_ABOVE_PREVAILING_HR','if PW_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR']
df_data_statistics.loc[['WAGE_RATE_OF_PAY_FROM'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','WAGE_RATE_OF_PAY_FROM_HR;WAGE_ABOVE_PREVAILING_HR','if WAGE_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR']

#Drop columns - Wage related
wage_cols=[ 'WAGE_UNIT_OF_PAY','PW_UNIT_OF_PAY']
df_data_statistics.loc[wage_cols,['preprocess_action','preprocess_comment']]=['Drop column','Not Useful']

#Feature engineering - OES_YN - if 'PW_OTHER_SOURCE' is not NaN then N else Y
df_data_statistics.loc[['PW_OTHER_SOURCE'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','OES_YN ','if PW_OTHER_SOURCE is not NaN then N else Y']
#Feature engineering - SURVEY_YEAR - if OES_YN ==Y then extract year from first date of PW_OES_YEAR' else 'PW_OTHER_YEAR'
df_data_statistics.loc[['PW_OES_YEAR','PW_OTHER_YEAR'],['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic']]=['Drop column','Feature engineering','SURVEY_YEAR ','if OES_YN ==Y then extract year from first date of PW_OES_YEAR else PW_OTHER_YEAR']

#Categorical columns 
cat_cols=['CASE_STATUS','VISA_CLASS','SOC_CODE','SOC_TITLE','EMPLOYER_NAME','EMPLOYER_POSTAL_CODE','WORKSITE_POSTAL_CODE','PW_OTHER_SOURCE','PUBLIC_DISCLOSURE','NAICS_CODE']
df_data_statistics.loc[cat_cols,['categorical_class', 'embedding']]=['Categorical','Standardized random']
#for employer name - append employer state and encode the combination
df_data_statistics.loc[['EMPLOYER_NAME'],['categorical_class', 'embedding']]=['Categorical','Standardized random for CONCAT(EMPLOYER_NAME,EMPLOYER_STATE)']

#Ordinal columns
ord_cols=['PW_WAGE_LEVEL','PW_OES_YEAR']
df_data_statistics.loc[ord_cols,['categorical_class', 'embedding']]=['Ordinal','Ordered standardized random']

#binary columns
binary_cols=['FULL_TIME_POSITION','AGENT_REPRESENTING_EMPLOYER','SECONDARY_ENTITY','AGREE_TO_LC_STATEMENT','H-1B_DEPENDENT','WILLFUL_VIOLATOR','EMPLOYER_COUNTRY']
df_data_statistics.loc[binary_cols,['categorical_class', 'embedding']]=['Binary','Standardized random']


#Save file
df_data_statistics.to_csv('/content/drive/MyDrive/preprocessing_steps_observations.csv',sep='$')