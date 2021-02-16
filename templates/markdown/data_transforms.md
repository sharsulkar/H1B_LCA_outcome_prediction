## PREPROCESSING PIPELINE
The preprocess pipeline is implemented based on the observations and suggestions from Exploratory Data Analysis discussed earlier. Sklearn's Pipeline, make_pipeline and make_column_transformer are used to build this preprocess pipeline.
The pipeline steps are described in detail below -  
**Load/Import data** -  
The data is sourced and loaded into a Pandas DataFrame object with only the necessary columns listed in the *required_features.csv* file imported. This ensures that the data given to the pipeline always has the same structure and order of columns.  
The preprocess pipeline is divided into 3 parts, where each part performs some transformation on the output of the previous part.  
### Pipeline part 1 - Drop rows and build features
**Step 1 - Drop rows where CASE_STATUS is not in ['Certified', 'Denied']** -  
This step performs below operations -  
a. Drop rows where CASE_STATUS is **not in** ['Certified', 'Denied']. We are not interested in predicting the outcomes other than ['Certified', 'Denied'] so we remove any rows that have CASE_STATUS other than the two we are interested in.  
b. Reset index after dropping the rows.  
*Please note that this step is applicable only for training, validation and testing datasets. The real time data used for prediction will not have CASE_STATUS field.*  

**Step 2 - Build Features** -  
This step builds new features based on the computation logic discussed in the previous section. The input to this transform is *feature_engineering_columns.csv* and the processed dataframe from output of step 1. 
This step computes new features and adds them as new columns to the end of it input dataframe. The dataframe is then passed on as the output of this step to the next step in pipeline.

### Pipeline part 2 - Separate and encode the target feature CASE_STATUS
**Step 1 - Separate the target feature CASE_STATUS** -  
The processed dataframe from part 1 of the pipeline still has the target feature present. This step separates it into its own series object.  

**Step 2 - Encode the target feature to binary** -  
The 2 class target feature is then encoded by a simple series replace transform as Confirmed -> 0, Denied -> 1.  

### Pipeline part 3 - Encoding numeric and categorical features
The below mentioned steps are implemented using a column_transformer that splits the processing of there steps and applies them for the given list of columns only.  
**Step 1 - Drop unnecessary features from processed dataframe from part 2**  
After the target feature is separated into its own series object, there are many features that were retained for building new features in part 1 which can be dropped now. The complete list of features that will be dropped is stored in *drop_columns.csv*. This step will drop the listed features inplace and return the dataframe will only the features required to train the model and make predictions. Currently this file has 31 features which are further divided into numerical and categorical features, as listed in *numeric_columns.csv* and *categorical_columns.csv* respectively. These will be processed separately in the next steps.

**Step 2 - Encode Categorical columns** -    
The column names stored in *categorical_columns.csv* are encoded using a custom built Random Standard Encoder which encodes each unique value in the categorical feature with a number randomly picked from a standard normal distribution. It is also compatible with the batch training approach with new categories and their encodings appended to what is already stored in memory. Check here for more details on its inner workings.  

**Step 3 - Encode Numeric columns** -  
The column names stored in *numeric_columns.csv* are encoded using a custom built standard scaler. Before scaling, the columns are passed through an imputer that **replaces missing values with the median** of the numeric column. The standard scaler is custom built to handle batch training approach as it uses pooled mean and variance in place of sample mean and variance to scale the data.  

The output of this part of the pipeline is then passed to the model to train, validate and predict.  
The part 1 and 3 of the pipeline is persisted in memory to be reused during batch training and to transform the test and prediction data.  


