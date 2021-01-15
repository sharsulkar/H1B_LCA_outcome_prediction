#import libraries
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from pickle import dump, load
from preprocessing_transforms import *
from new_feature_transforms import *

#import observations_df for referencing features and corresponding preprocessing actions to be performed on them
#observations_df=pd.read_csv('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/reports/final_observations.csv',sep='$',index_col=0,error_bad_lines=False)

def read_csv_to_list(filepath,header=None,squeeze=True):
  return list(pd.read_csv(filepath,header=None,squeeze=True))

#identify and define column sets for applying preprocessing transforms
num_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/numeric_columns.csv',header=None,squeeze=True)
cat_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/categorical_columns.csv',header=None,squeeze=True)
drop_cols=read_csv_to_list('https://github.com/sharsulkar/H1B_LCA_outcome_prediction/raw/main/data/processed/drop_columns.csv',header=None,squeeze=True)
fe_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
required_features=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/required_features.csv',header=None,squeeze=True)

#source data 
input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx',usecols=required_features)
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

print(X.shape)
#save transformed dataset and target
#pd.DataFrame(X,columns=fe_df.columns.values).to_csv('/content/drive/MyDrive/Datasets/processed.csv')

#save pipeline
#reference - https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
#dump(all_preprocess,open('/content/drive/MyDrive/preprocess_pipe.pkl','wb'))
#all_preprocess=load(open('/content/drive/MyDrive/preprocess_pipe.pkl','rb'))