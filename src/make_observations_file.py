import numpy as np
import pandas as pd
import logging
from mylib import read_csv_to_list, modify_observations, missing_statistics, cardinality_statistics
from transforms import BuildFeaturesTransformer, DropRowsTransformer
from sklearn.pipeline import Pipeline, make_pipeline


logging.info('Starting the program to recreate the observations file.')
#load data
input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx')

#build features
fe_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
drop_row_index=input_df[~input_df.CASE_STATUS.isin(['Certified','Denied'])].index

#Build preprocessing pipeline
build_feature_pipe=make_pipeline(
    DropRowsTransformer(row_index=drop_row_index,inplace=True,reset_index=True),
    BuildFeaturesTransformer(fe_cols)
)

#apply preprocessing pipeline
transformed_df=build_feature_pipe.transform(input_df)

#Recreate observations file
#store feature statictics in a dataframe
observations_df=pd.DataFrame(data=None,
                            index=transformed_df.columns,
                            columns=['Dtype','percent_missing','cardinality','preprocess_action','preprocess_comment','new_feature_name','new_feature_logic','categorical_class','embedding']
)

#Fill in Dtype, missing and cardinality statistics
for column in transformed_df.columns:
  
  observations_df=modify_observations(df=observations_df,
                                      index=column,
                                      columns=['Dtype','percent_missing','cardinality'],
                                      values=[transformed_df[column].dtype,missing_statistics(transformed_df,column),cardinality_statistics(transformed_df,column)],
                                      modify_action='update_values')

#mark features with missing values >threshold for dropping
missing_threshold=40.0
for idx in observations_df[observations_df.percent_missing>=missing_threshold].index:
  observations_df=modify_observations(df=observations_df,
                                      index=idx,
                                      columns=['preprocess_action','preprocess_comment'],
                                      values=['Drop column','missing values>='+str(missing_threshold)+'% of total'],
                                      modify_action='update_values')

#mark features with cardinality >threshold for dropping
cardinality_threshold=80.0
for idx in observations_df[observations_df.cardinality<80.0].index:
  observations_df=modify_observations(df=observations_df,
                                      index=idx,
                                      columns=['preprocess_action','preprocess_comment'],
                                      values=['Drop column','High Cardinality, threshold '+str(cardinality_threshold)+'% of total'],
                                      modify_action='update_values')

#mark target column
observations_df=modify_observations(df=observations_df,
                                    index='CASE_STATUS',
                                    columns=['preprocess_action','preprocess_comment'],
                                    values=['Pop column into a separate list','Target feature'],
                                    modify_action='update_values')

#FEATURE Engineering - date columns
#Create a new feature - PROCESSING_DAYS from 'RECEIVED_DATE', 'DECISION_DATE'
observations_df=modify_observations(df=observations_df,
                                    index=['RECEIVED_DATE', 'DECISION_DATE'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','PROCESSING_DAYS','days(DECISION_DATE-RECEIVED_DATE)'],
                                    modify_action='update_values')

#Create a new feature - VALIDITY_DAYS from 'BEGIN_DATE', 'END_DATE'
observations_df=modify_observations(df=observations_df,
                                    index=['BEGIN_DATE', 'END_DATE'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','VALIDITY_DAYS','days(END_DATE-BEGIN_DATE)'],
                                    modify_action='update_values')

#Feature engineering - split SOC_CODE into 2 new features - SOC_CODE_2, SOC_CODE_4
observations_df=modify_observations(df=observations_df,
                                    index='SOC_CODE',
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','SOC_CODE_2,SOC_CODE_4','SOC_CODE.split(\'-\')'],
                                    modify_action='update_values')

#Feature engineering - EMPLOYER_COUNTRY - US or NOT
observations_df=modify_observations(df=observations_df,
                                    index='EMPLOYER_COUNTRY',
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','USA_YN','IF EMPLOYER_COUNTRY==USA THEN Y ELSE N END'],
                                    modify_action='update_values')

#Drop columns - EMPLOYER_* except 'EMPLOYER_NAME',EMPLOYER_POSTAL_CODE
not_useful_cols=['TRADE_NAME_DBA','EMPLOYER_ADDRESS1','EMPLOYER_ADDRESS2','EMPLOYER_CITY','EMPLOYER_STATE',
          'EMPLOYER_PROVINCE','EMPLOYER_PHONE','EMPLOYER_PHONE_EXT','EMPLOYER_POC_LAST_NAME',
          'EMPLOYER_POC_FIRST_NAME','EMPLOYER_POC_MIDDLE_NAME','EMPLOYER_POC_JOB_TITLE','EMPLOYER_POC_ADDRESS1',
          'EMPLOYER_POC_ADDRESS2','EMPLOYER_POC_CITY','EMPLOYER_POC_STATE','EMPLOYER_POC_POSTAL_CODE',
          'EMPLOYER_POC_COUNTRY','EMPLOYER_POC_PROVINCE','EMPLOYER_POC_PHONE','EMPLOYER_POC_PHONE_EXT','EMPLOYER_POC_EMAIL',
          'AGENT_ATTORNEY_LAST_NAME','AGENT_ATTORNEY_FIRST_NAME','AGENT_ATTORNEY_MIDDLE_NAME','AGENT_ATTORNEY_ADDRESS1',
          'AGENT_ATTORNEY_ADDRESS2','AGENT_ATTORNEY_CITY','AGENT_ATTORNEY_STATE','AGENT_ATTORNEY_POSTAL_CODE',
          'AGENT_ATTORNEY_COUNTRY','AGENT_ATTORNEY_PROVINCE','AGENT_ATTORNEY_PHONE','AGENT_ATTORNEY_PHONE_EXT',
          'AGENT_ATTORNEY_EMAIL_ADDRESS','LAWFIRM_NAME_BUSINESS_NAME','STATE_OF_HIGHEST_COURT','NAME_OF_HIGHEST_STATE_COURT','SECONDARY_ENTITY_BUSINESS_NAME',
          'WORKSITE_ADDRESS1','WORKSITE_ADDRESS2','WORKSITE_CITY','WORKSITE_COUNTY','WORKSITE_STATE','WAGE_UNIT_OF_PAY','PW_UNIT_OF_PAY','APPENDIX_A_ATTACHED','STATUTORY_BASIS']
observations_df=modify_observations(df=observations_df,
                                    index=not_useful_cols,
                                    columns=['preprocess_action','preprocess_comment'],
                                    values=['Drop column','Not Useful'],
                                    modify_action='update_values')

#Feature engineering - Worksite same as employer address 
observations_df=modify_observations(df=observations_df,
                                    index=['WORKSITE_POSTAL_CODE','EMPLOYER_POSTAL_CODE'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','EMPLOYER_WORKSITE_YN','IF EMPLOYER_POSTAL_CODE==WORKSITE_POSTAL_CODE THEN Y ELSE N END'],
                                    modify_action='update_values')

#Feature engineering - convert PREVAILING_WAGE and WAGE_RATE_OF_PAY_FROM to hourly wage - if PW_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067
#Feature engineering - WAGE_ABOVE_PREVAILING_HR = WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR
observations_df=modify_observations(df=observations_df,
                                    index=['PREVAILING_WAGE','PW_UNIT_OF_PAY'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','PREVAILING_WAGE_HR;WAGE_ABOVE_PREVAILING_HR','if PW_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR'],
                                    modify_action='update_values')

observations_df=modify_observations(df=observations_df,
                                    index=['WAGE_RATE_OF_PAY_FROM','WAGE_UNIT_OF_PAY'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','WAGE_RATE_OF_PAY_FROM_HR;WAGE_ABOVE_PREVAILING_HR','if WAGE_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR'],
                                    modify_action='update_values')

#Feature engineering - OES_YN - if 'PW_OTHER_SOURCE' is not NaN then N else Y
observations_df=modify_observations(df=observations_df,
                                    index='PW_OTHER_SOURCE',
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','OES_YN ','if PW_OTHER_SOURCE is not NaN then N else Y'],
                                    modify_action='update_values')

#Feature engineering - SURVEY_YEAR - if OES_YN ==Y then extract year from first date of PW_OES_YEAR' else 'PW_OTHER_YEAR'
observations_df=modify_observations(df=observations_df,
                                    index=['PW_OES_YEAR','PW_OTHER_YEAR'],
                                    columns=['preprocess_action','preprocess_comment','new_feature_name','new_feature_logic'],
                                    values=['Drop column','Feature engineering','SURVEY_YEAR ','if OES_YN ==Y then extract year from first date of PW_OES_YEAR else PW_OTHER_YEAR'],
                                    modify_action='update_values')

#Categorical columns 
cat_cols=['CASE_STATUS','VISA_CLASS','SOC_CODE','SOC_TITLE','EMPLOYER_NAME','EMPLOYER_POSTAL_CODE','WORKSITE_POSTAL_CODE','PW_OTHER_SOURCE','PUBLIC_DISCLOSURE','NAICS_CODE','EMPLOYER_NAME']
observations_df=modify_observations(df=observations_df,
                                    index=cat_cols,
                                    columns=['categorical_class', 'embedding'],
                                    values=['Categorical','Standardized random'],
                                    modify_action='update_values')

#Ordinal columns
ord_cols=['PW_WAGE_LEVEL','PW_OES_YEAR']
observations_df=modify_observations(df=observations_df,
                                    index=ord_cols,
                                    columns=['categorical_class', 'embedding'],
                                    values=['Ordinal','Ordered standardized random'],
                                    modify_action='update_values')

#binary columns
binary_cols=['FULL_TIME_POSITION','AGENT_REPRESENTING_EMPLOYER','SECONDARY_ENTITY','AGREE_TO_LC_STATEMENT','H-1B_DEPENDENT','WILLFUL_VIOLATOR','EMPLOYER_COUNTRY']
observations_df=modify_observations(df=observations_df,
                                    index=binary_cols,
                                    columns=['categorical_class', 'embedding'],
                                    values=['Binary','Standardized random'],
                                    modify_action='update_values')

#numeric columns
numeric_cols=['TOTAL_WORKER_POSITIONS', 'NEW_EMPLOYMENT', 'CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT', 'NEW_CONCURRENT_EMPLOYMENT','CHANGE_EMPLOYER', 'AMENDED_PETITION', 'WORKSITE_WORKERS','TOTAL_WORKSITE_LOCATIONS']
observations_df=modify_observations(df=observations_df,
                                    index=numeric_cols,
                                    columns=['categorical_class', 'embedding'],
                                    values=['Numerical','Standard scaling'],
                                    modify_action='update_values')

#Update details for new features - Numeric
observations_df=modify_observations(observations_df,
                                    index=['PROCESSING_DAYS','VALIDITY_DAYS','WAGE_ABOVE_PW_HR'],
                                    columns=['preprocess_action','preprocess_comment','categorical_class','embedding'],
                                    values=['New feature','Feature engineering','Numerical','Standard scaling'],
                                    modify_action='update_values')

##Update details for new features - Categorical
observations_df=modify_observations(observations_df,
                                    index=['SOC_CD2','SOC_CD4','SOC_CD_ONET'],
                                    columns=['preprocess_action','preprocess_comment','categorical_class','embedding'],
                                    values=['New feature','Feature engineering','Categorical','Standardized random'],
                                    modify_action='update_values')

##Update details for new features - Binary
observations_df=modify_observations(observations_df,
                                    index=['USA_YN','EMPLOYER_WORKSITE_YN','OES_YN'],
                                    columns=['preprocess_action','preprocess_comment','categorical_class','embedding'],
                                    values=['New feature','Feature engineering','Binary','Standardized random'],
                                    modify_action='update_values')

##Update details for new features - Ordinal
observations_df=modify_observations(observations_df,
                                    index=['SURVEY_YEAR'],
                                    columns=['preprocess_action','preprocess_comment','categorical_class','embedding'],
                                    values=['New feature','Feature engineering','Ordinal','Ordered standardized random'],
                                    modify_action='update_values')

#features that will be used without any modifications
observations_df=modify_observations(observations_df,
                                    index=['VISA_CLASS', 'SOC_TITLE', 'FULL_TIME_POSITION','TOTAL_WORKER_POSITIONS', 'NEW_EMPLOYMENT', 'CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT', 'NEW_CONCURRENT_EMPLOYMENT','CHANGE_EMPLOYER', 'AMENDED_PETITION', 'EMPLOYER_NAME', 'NAICS_CODE','AGENT_REPRESENTING_EMPLOYER', 'WORKSITE_WORKERS', 'SECONDARY_ENTITY','PW_WAGE_LEVEL', 'TOTAL_WORKSITE_LOCATIONS', 'AGREE_TO_LC_STATEMENT','H-1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'PUBLIC_DISCLOSURE'],
                                    columns=['preprocess_action','preprocess_comment'],
                                    values=['Use feature as is','Use feature as is'],
                                    modify_action='update_values')

#Save observations file
observations_df.to_csv('/data/processed/preprocessing_steps_observations.csv',sep='$')

