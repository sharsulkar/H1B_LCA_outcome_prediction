# Data Analysis -  
## Problem statement  
**Background** - DHS programs for speciality occupation need to undergo Department of Labor (DOL) Certification to ensure the local workforce is not adverseley impacted due to 
jobs offered to foreign workforce. LCA and PERM certifications are part of the DOL certifications legally required before the work application for foreign labor is approved.  
This project will try to model the underlying process and provide solutions to below mentioned aspects of this process -  
1. Exploratory Data Analysis to answer questions mentioned in ./EDA_viz.md file.  
2. Predict the outcome of a given application under the OFLC LCA or PERM programs (Separate models for each program).  
3. Given the prediction, suggest what changes to the application parameters will most likely change the outcome.  
4. Given the application parameters, what is the closest case-number in the last 3 years display its outcome.

## Source data info 
Data will be sourced from publicly disclosed data on the below website -  
<span style="color:green">Reference</span> https://www.dol.gov/agencies/eta/foreign-labor/performance  

## Data Observations for LCA based programs (Separate but similar steps needed for PERM program)  
### Preprocessing recommendations  
1. Reduce Dataframe space by using appropriate datatypes. Only import columns that are required.
2. The output class is unbalanced. Consider it during training, selecting evaluation metrics.
3. Missing values - Drop columns that have more than 40% missing values.
4. Cardinality - Drop columns that have more than 80% of its total values as unique.
5. Drop rows that are not with CASE_STATUS Certified or Denied.  

### Feature Engineering recommendations for LCA based programs (Separate but similar steps neede for PERM program)
<span style="color:green">Reference</span>- https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Record_Layout_FY2020.pdf  
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
### Supporting visualizations  

## Preprocessing transforms -  
### Transformation Pipeline  -  
1. Import raw data into pandas Dataframe  
    <span style="color:orange">test</span>  - Column names match expected column names to detect format changes  
    <span style="color:yellow">logging</span>  - file size in MBs, dataframe shape  
2. Transform columns to engineer features  
    <span style="color:orange">test</span> - test transform functionality, Dtypes  
    <span style="color:yellow">logging</span>  - status when transforming is complete  
3. Drop features identified as redundant  
    <span style="color:orange">test</span> - before dropping -  
    a. features that are being dropped are present in DF  
    b. feature transforms dependant on them are generated  
    after dropping - features are no longer present  
    <span style="color:yellow">logging</span>  - list of features dropped with status  
4. Encode non-numeric features to numerical embeddings and impute missing values  
    <span style="color:orange">test</span> - encoding functionality, no non-numeric data after encoding is complete, no missing values  
    <span style="color:yellow">logging</span>  - status when transforming is complete  
5. Standardize numeric features  
    <span style="color:orange">test</span> - encoding functionality, standardization, no missing values  
    <span style="color:yellow">logging</span> - status when transforming is complete   
6. Convert to appropriate Dtypes for storage efficiency  
    <span style="color:orange">test</span> - Dtypes after transforming is complete  
    <span style="color:yellow">logging</span>  - file size in MBs after conversion   
7. export processed dataset, embeddings and parameters for next steps, future use  
    <span style="color:orange">test</span> - functionality, file exists in expected path after exporting  
    <span style="color:yellow">logging</span>  - file path where file is exported  
### Supporting diagrams  
