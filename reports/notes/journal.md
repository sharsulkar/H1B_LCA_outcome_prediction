## Journal to store day-to-day notes, important observations, thoughts and misc.  

### 1/7/21 -  
#### approx time spent - 2.5h  
Continued working on coding the core functionality in python for **LCA data**. Almost finished coding the core functionality of the feature engineering steps, still 1-2 features have to be implemented.  


### 1/6/21 -  
#### approx time spent - 3h  
Today, working on prototyping the core functionality for **LCA data** for preprocessing data transforms noted in /notes/data_transforms.md. Also note down any additional preprocessing steps and testing or logging steps that 
come to mind. Will need to do similar steps for PERM data later.  
Instead of loading the entire dataset and then dropping columns that are not needed, it might be better to just load the required columns and set their datatypes during import. Dropping the columns that will be used for feature engineering 
will be needed but atleast the list of columns to drop will be comparatively less.  
Thinking of implementing the pipeline in below order -  
1. Select required features and import with correct dtypes
On filtering columns before import, the size of imported data frame reduced by >50%. On further thought and reading around, it might be ok to skip specifying correct Dtypes during import and all features are eventually going to be numeric at
the end of the data_transforms step.
2. Drop rows where CASE_STATUS not in ('Certified','Denied')
3. Do feature engineering steps
4. drop columns
5. embed categorical columns, normalize numerical columns
6. pop the target column into a separate list

### 1/5/21 -
#### approx time spent - 3h  
After some thought and analysis, it seems like this problem is much better represented as anomoly detection than a multiclass classification.
On an avg, 90-95% of all LCA and PERM applications are Certified while less than 1% are denied. It will be in the interest of applicants to 
know if their application is likely to be denied and if so what can they do to avoid that outcome.  
The other 2 outcomes that have 2-4% probability are related to withdrawals which are initiated by the applicants so predicting them is not necessary 
or helpful as they lie outside the decision making framework of US Department of Labor (in a way, certified-withdrawn has already been certified but 
still withdrawn although that just increases the certified count so my thought process still holds true).  

First look at the data does not show any glaring characteristics of denied applications (which is good else this project would have ended here).
A few things on the top of my head that can be checked are denial rates on -  
1. non-H-1B applications
2. any specific SOC titles having more denial rates
3. prevailing wage vs wage offered
4. Fulltime position - Y/N
5. Willful violaters/H1B dependent employers
I will also have to check if any findings from above are same over different years or not.  

On checking denial rates for above 5 conditions, I did not find anything conclusive that can explain denials, 
all of the above have Certified as well as Denied rows. So it is worth putting this data through a ML algorithm to see if it can pick up any patterns.

Another observation - Each year's LCA format is bit different which will have to be considered when doing preprocessing and training for 2019 and before.
Although the key feature columns tend to remain the same.  

#### Action items - done  
--Add preprocessing step to remove rows that are not with CASE_STATUS Certified or Denied  
LCA_df.drop(LCA_df[LCA_df.CASE_STATUS.isin(['Certified - Withdrawn', 'Withdrawn'])].index,inplace=True)  

### Before 1/1/21 -  
#### approx time spent - 10h
Worked on identifying preprocessing steps for LCA data. Detailed result and observations are in /reports/notes/data_transforms.md.  
prototype code is in /prototyping/notebooks/01_sh_EDA.ipynb.  





