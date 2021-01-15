## Journal to store day-to-day notes, important observations, thoughts and misc.  
### 1/15/21 -
#### approx time spent - 5h 
Moved the preprocessing transformer classes to seperate files that can be reused and imported.  
The code to generate the observations_df and columns can be part of the make_dataset.py file as it is in a way essential to processing the data. 
Based on the how the training and test dataset should be built, that code will also be part of this file. 

Today I want to finish below tasks -
1. refactoring all preprocessing code, make preprocess transform classes reusable
2. write separate code for rebuilding observations_df and saving the columns
3. note down test cases
4. documentation - doc strings

### 1/13/21 -
#### approx time spent - 5h 
The training and prediction pipelines will have to be different. Although the preprocess steps are exactly same, the training pipeline will have additional steps of removing rows where CASE_STATUS.notin('Certified','Denied') and separating the target column. 
During building the pipeline i am getting an error for new features saying they are missing from the dataframe, which is correct but the intention of the pipeline is to execute steps in given order. The features will be available when the feature_engineering step is executed. Instead of spending time trying to fix this issue, I am more inclined to skip using the pipeline and add all the transform logic into build_features.py file. In a way, it will work as a pipeline too.
Nevermind my above comment, I am able to use the pipeline feature by splitting the feature engineering step into its own pipeline and applying that to the dataframe before hand.

Pickled and saved the pipeline object. Saved the transformed dataset on google drive. I dont know if github will allow to store datasets >50MB, else google drive might be a good place and I will just store the link to dataset in github.

I think I will write a separate code to generate the observations_df (not necessarily refactored code, just a cleaned version of prototype code is fine). /reports/ will be a good place to store it and showcase it as part of the final report. I also think it will be a good idea to persist the various set of columns as they will remain constant once finalized (eg numeric, categorical, feature engineering, required, drop features). /data/processed folder will be a good place to store them.

**How to write better prototyping code so refactoring takes as little time as possible**  
1. Use Pep8 style and appropriate names that can be carried forward in the final code without change
2. Add comments where needed
3. Add notes on tests, error and exception handling in the prototype code
4. Code for better performance in production

### 1/12/21 -
#### approx time spent - 4h 
Separated the pipeline prototype code into its own notebook. The code depeloped to build feature will be used in the pipeline so the output of 02_sh_build_features that I care about is actually the observations_df, so saved it as final_observations.csv with a $ delimiter. While refactoring, I will also need to create a code to generate the observations_df in case the file generate during protoyping is deleted.  

For categorical encoding, I am using the custom built random standard_normal encoder because the sklearn encoders like onehotencoder, labelencoder do not serve my purpose. I will explain more in the detailed notes. One functionality that I will have to work on for this custom encoder is a notification mechanism when categorical values that are not part of training encoding are found during the transform. Like any other categorical encoder, this will fail so an early notification will be helpful. In addition to that, I will need to think on how to handle when this happens. 

Today I want to finish the below tasks -
1. Encoding all features - done
2. Class for feature engineering that can go into the pipeline - done
3. Create a prototype feature transform pipeline - almost done
4. Save the processed dataset, encoding parameters and pipeline object
5. Start thinking on steps for refactoring and documentation.

### 1/11/21 -
#### approx time spent - 4h 
Finished the code for all feature engineering. Worked on streamlining and updating the observations_df to include newly engineered features. Started working on encoding categorical and numerical features and building the pipeline. 

### 1/8/21 -
#### approx time spent - 3h 
Sidenote - Gitpod.io has a 50hr/month usage limit for free mode so did not want to waste it on editing notes and documents. Found and moved to dillinger.io for editing .md documents for this project. Looks like a simple online md editor which I can use from anywhere to edit and syncup notes and documents from github, will give it a try and hope it works.  

Finished coding for SURVEY_YEAR and WAGE_ABOVE_PW_HR features. Before moving to feature encoding, I think it is better to add/update the preprocessing_steps_observations.csv filling in details with the new features engineered and also what type of encoding will suit best. This way, the preprocessing_steps_observations.csv can be a central location to record and reuse all actions taken in the preprocessing steps.  

##### Today I Learned -
1. Dataframe column can be updated using a series object using df.update(series) if the series and column name to be updated have the same name.
2. Pandas has a to_datetime method to convert a column into datetime. individual parts of datetime can then be extracted using the .dt.tp_period('D'/'M'/'Y').
3. pd.apply only works if applied on a single df column. for more than 1 columns, its easier to update index-wise.

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





