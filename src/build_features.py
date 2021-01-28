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
module_logger= logging.getLogger('my_application.build_features')

def main(input_df):

    module_logger.info('Starting to build features.')

    #identify and define column sets for applying preprocessing transforms
    num_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/numeric_columns.csv',header=None,squeeze=True)
    cat_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/categorical_columns.csv',header=None,squeeze=True)
    drop_cols=read_csv_to_list('https://github.com/sharsulkar/H1B_LCA_outcome_prediction/raw/main/data/processed/drop_columns.csv',header=None,squeeze=True)
    fe_cols=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
        
    module_logger.info('Importing columns from stored lists complete.')

    drop_row_index=input_df[~input_df.CASE_STATUS.isin(['Certified','Denied'])].index
    module_logger.info('Number of rows with CASE_STATUS other than Certified and Denied:%d',drop_row_index.shape[0])

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
    module_logger.info('Pipeline started')
    #feature engineering + drop rows
    fe_df=build_feature_pipe.fit_transform(input_df)
    #Ensure that there are no records with CASE_STATUS not in ['Certified','Denied'] after drop_rows  
    assert fe_df[~fe_df.CASE_STATUS.isin(['Certified','Denied'])].shape[0]==0, module_logger.error('Unexpected values found in CASE_STATUS field.')

    module_logger.info('feature engineering + drop rows done')

    #Separate target column - add conditions to apply only on training dataset
    y=fe_df.pop('CASE_STATUS')
    module_logger.info('Target column separated')

    #drop columns + encoding
    X=all_preprocess.fit_transform(fe_df)

    #Ensure that X has expected number of features  
    assert X.shape[1]==30,module_logger.exception('Arrays X of shape [:,31] expected.')
    #Ensure that X and y have same number of rows  
    assert X.shape[0]==y.shape[0],module_logger.exception('Arrays X and y should have same number of rows.') 
    module_logger.info('drop columns + encoding done')

    #save transformed dataset and target 
    #pd.DataFrame(X,columns=fe_df.columns.values).to_csv('/content/drive/MyDrive/Datasets/processed.csv') 
    module_logger.info('Building features complete.')
    #save pipeline
    #dump(all_preprocess,open('/content/drive/MyDrive/preprocess_pipe.pkl','wb')) 
    ##all_preprocess=load(open('/content/drive/MyDrive/preprocess_pipe.pkl','rb')) 
    return X, y

if __name__ == '__main__':
    import make_dataset

    input_df=make_dataset.main()
    main(input_df)