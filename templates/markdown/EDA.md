## RECOMMENDATIONS FROM EXPLORATORY DATA ANALYSIS 
Below actionalble recommendations were derived after performing EDA on the training data. They are implemented in the preprocessing pipeline, which is discussed in detail in the next sections.  
The entire list of recommendations and observations for each source field is available here -    https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/reports/preprocessing_steps_observations.html  
### RECOMMENDATION FOR PREPROCESSING TASKS   
1. Reduce Dataframe space by using appropriate datatypes. Only import columns that are required.  
The list of these columns are available in the Github repository at ./data/processed/required_features.csv.  
2. The output class is unbalanced. Consider it during training, selecting evaluation metrics.  
3. Missing values - **Drop columns** that have more than 40% missing values.  
4. Cardinality - **Drop columns** that have more than 80% of its total values as unique.  
5. **Drop rows** that are not with CASE_STATUS Certified or Denied.  
 
### POPULATE TASK-SPECIFIC FEATURE LIST
The below list of columns are an outcome of the EDA observations are persisted in memory in CSV format and will be used for different preprocessing tasks. They are available in the /data/processed/ folder in the code repository.  
1. **required_features** - list of columns that will be limit which fields from the source data will be imported. The source data consists of 96-100 fields, not all of which are useful.  
2. **categorical_columns** - list of columns that have been identified as containing categorical datatypes, which include nominal, ordinal and binary.  
3. **numeric_columns** - list of columns that have been identified as containing numerical data, which include integers, floats and timedelta.  
4. **drop_columns** - list of columns that have been identified as not needed in the training and prediction operations. They are dropped after the feature engineering step.  
5. **feature_engineering_columns** - list of columns that will be used to build new features. The list of new features and the computational logic is as given below.  

### FEATURE ENGINEERING RECOMMENDATIONS   
Generate new features mentioned below from features available in source data.  
1. **PROCESSING_DAYS**   
**Computation** days(DECISION_DATE - RECIEVED_DATE)  
**Reason** - The number of days required to process an application can be used as a feature, especially if the outcome is correlated with any delays in processing.  
This will replace the features - DECISION_DATE, RECIEVED_DATE.  
2. **VALIDITY_DAYS**  
**Computation** - days(END_DATE - BEGIN_DATE)  
**Reason** - The number of days work authorization is requested can be used as a feature, especially if the outcome is correlated with any delays in processing.  
This will replace the features - DECISION_DATE, RECIEVED_DATE.  
3. **SOC_CODE_2, SOC_CODE_4, SOC_CODE_ONET**  
<span style="color:green">Reference</span> - https://en.wikipedia.org/wiki/Standard_Occupational_Classification_System  
**Computation** - SOC_CODE_2=SOC_CODE.split(\'-\')[0];  
SOC_CODE_4=SOC_CODE.split(\'-\')[1];  
SOC_CODE_ONET=SOC_CODE.split(\'.\')[1];  
**Reason** - SOC_CODE identifes the job being requested and is of the format AA_BBBB.CC. Splitting the various parts of the code might help tie outcomes to specific job roles, industries or specializations. Also encoding might be easier.  
4. **EMPLOYER_COUNTRY** -  
**Computation** - IF EMPLOYER_COUNTRY==USA THEN Y ELSE N END  
**Reason** - the country is almost always USA but very rarely not USA so this feature converts the Employer_country to a binary feature.  
5. **PREVAILING_WAGE_HR** -  
**Computation** - if PW_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR  
**Reason** - Standardize the wage rate to per Hour  
6. **WAGE_RATE_OF_PAY_FROM_HR** -  
**Computation** - if WAGE_UNIT_OF_PAY=Hour ignore, if Month then WAGE/172, if Year then WAGE/2067;WAGE_RATE_OF_PAY_FROM_HR-PREVAILING_WAGE_HR  
**Reason** - Standardize the wage rate to per Hour  
7. **OES_YN** -  
**Computation** - if PW_OTHER_SOURCE is not NaN then N else Y  
**Reason** - Whether OES was used for prewailing wage survey or not   
8. **SURVEY_YEAR** -  
**Computation** - if OES_YN ==Y then extract year from first date of PW_OES_YEAR else PW_OTHER_YEAR  
**Reason** - The year of survey used.  
