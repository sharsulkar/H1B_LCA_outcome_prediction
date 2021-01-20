#import libraries
import numpy as np
np.random.seed(42)
import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from pickle import dump, load
import transforms 
from mylib import read_csv_to_list

#import observations_df for referencing features and corresponding preprocessing actions to be performed on them
#observations_df=pd.read_csv('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/reports/final_observations.csv',sep='$',index_col=0,error_bad_lines=False)


def main(input_df):
    #identify and define column sets for applying preprocessing transforms
    num_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/numeric_columns.csv',header=None,squeeze=True)
    cat_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/categorical_columns.csv',header=None,squeeze=True)
    drop_cols=read_csv_to_list('https://github.com/sharsulkar/H1B_LCA_outcome_prediction/raw/main/data/processed/drop_columns.csv',header=None,squeeze=True)
    fe_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
        
    logging.info('Importing columns from stored lists complete.')
    #logging.info('Numeric columns:%s',num_cols)
    #logging.info('Categorical columns:%s',cat_cols)
    #logging.info('Columns to be dropped:%s',drop_cols)
    #logging.info('Columns used for feature engineering:%s',fe_cols)
    #logging.info('Required columns:',required_features)

    #source data 
    #input_df=make_dataset.main()
    #logging.info('Input dataframe imported with shape:',input_df.shape)

    drop_row_index=input_df[~input_df.CASE_STATUS.isin(['Certified','Denied'])].index
    logging.info('Number of rows with CASE_STATUS other than Certified and Denied:%s',drop_row_index.shape[0])

    #Build preprocessing pipeline
    build_feature_pipe=make_pipeline(
        transforms.DropRowsTransformer(row_index=drop_row_index,inplace=True,reset_index=True),
        transforms.BuildFeaturesTransformer(fe_cols)
        )

    numerical_preprocess=make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    preprocess_pipe=make_column_transformer(
        (transforms.DropFeaturesTransformer(columns=list(drop_cols),inplace=True),list(drop_cols)),
        (transforms.RandomStandardEncoderTransformer(cat_cols),cat_cols),
        (numerical_preprocess,num_cols),
        remainder='passthrough'
    )

    all_preprocess=make_pipeline(
        preprocess_pipe
    )


    #apply pipeline
    logging.info('Pipeline started')
    #feature engineering + drop rows
    fe_df=build_feature_pipe.fit_transform(input_df)
    logging.info('feature engineering + drop rows done')
    #Separate target column - add conditions to apply only on training dataset
    y=fe_df.pop('CASE_STATUS')
    logging.info('Target column separated')
    #drop columns + encoding
    X=all_preprocess.fit_transform(fe_df)
    logging.info('drop columns + encoding done')
    #save transformed dataset and target 
    #pd.DataFrame(X,columns=fe_df.columns.values).to_csv('/content/drive/MyDrive/Datasets/processed.csv') 

    #save pipeline
    #dump(all_preprocess,open('/content/drive/MyDrive/preprocess_pipe.pkl','wb')) 
    ##all_preprocess=load(open('/content/drive/MyDrive/preprocess_pipe.pkl','rb')) 
    return X, y

if __name__ == '__main__':
    main()