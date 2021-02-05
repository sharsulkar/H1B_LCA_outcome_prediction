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


module_logger= logging.getLogger('my_application.build_features')

def main(input_df, build_feature_pipe=None, all_preprocess=None, method='fit_transform'):
    

    module_logger.info('Starting to build features module.')

    #identify and define column sets for applying preprocessing transforms
    num_cols=read_csv_to_list('./data/processed/numeric_columns.csv',header=None,squeeze=True)
    cat_cols=read_csv_to_list('./data/processed/categorical_columns.csv',header=None,squeeze=True)
    drop_cols=read_csv_to_list('./data/processed/drop_columns.csv',header=None,squeeze=True)
    fe_cols=read_csv_to_list('./data/processed/feature_engineering_columns.csv',header=None,squeeze=True)
        
    module_logger.info('Importing columns from stored lists complete.')

    #Build preprocessing pipeline 
    if method in ['fit','fit_transform'] and build_feature_pipe is None:
        module_logger.info('Building build_features_pipe for very first time.')
        build_feature_pipe=make_pipeline(
            transforms.DropRowsTransformer(),
            transforms.BuildFeaturesTransformer(fe_cols)
            )

    if method in ['fit','fit_transform'] and all_preprocess is None:
        module_logger.info('Building all_preprocess_pipe for very first time.')

        numerical_preprocess=make_pipeline(
            SimpleImputer(strategy='median'),
            transforms.CustomStandardScaler()
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

    if method=='fit':
        module_logger.info('Starting pipeline.fit')

        fe_df=build_feature_pipe.fit_transform(input_df) 
        all_preprocess.fit(fe_df)
        
        X=[] #return empty array as only pipeline is fitted
        y=[] #return empty array as only pipeline is fitted
        module_logger.info('Pipeline.fit completed successfully')

    elif method=='transform':

        module_logger.info('Starting pipeline.transform')

        #build_feature_pipe, all_preprocess cannot be None
        assert build_feature_pipe!=None, module_logger.error('Missing pipeline object build_feature_pipe.')
        assert all_preprocess!=None, module_logger.error('Missing pipeline object all_preprocess.')

        fe_df=build_feature_pipe.transform(input_df) 
        module_logger.info('feature engineering + drop rows done')

        #Check if input_df has column CASE_STATUS, dataset during prediction will not have target variable so below code should be skipped
        if 'CASE_STATUS' in fe_df.columns.values: 
            module_logger.info('Target column found.')

            assert fe_df[~fe_df.CASE_STATUS.isin(['Certified','Denied'])].shape[0]==0, module_logger.error('Unexpected values found in CASE_STATUS field.')
            y=fe_df.pop('CASE_STATUS')
            y.replace(['Certified','Denied'],[0,1],inplace=True)

            module_logger.info('Target column separated')
        #if CASE_STATUS is not present, return empty array for y
        else:
            module_logger.info('Target column not found. Returning empty array for y.')
            y=[]

        X=all_preprocess.transform(fe_df)

        #Ensure that X has expected number of features  
        assert X.shape[1]==31,module_logger.exception('Arrays X of shape [:,31] expected.')
        #Ensure that X and y have same number of rows  
        assert X.shape[0]==y.shape[0],module_logger.exception('Arrays X and y should have same number of rows.') 

        module_logger.info('drop columns + encoding done')
        module_logger.info('pipeline.transform completed successfully')

    elif method=='fit_transform':
        module_logger.info('Starting pipeline.fit_transform')
        fe_df=build_feature_pipe.fit_transform(input_df) 
        module_logger.info('feature engineering + drop rows done')

        assert fe_df[~fe_df.CASE_STATUS.isin(['Certified','Denied'])].shape[0]==0, module_logger.error('Unexpected values found in CASE_STATUS field.')
        y=fe_df.pop('CASE_STATUS')
        y.replace(['Certified','Denied'],[0,1],inplace=True)
        module_logger.info('Target column separated')

        X=all_preprocess.fit_transform(fe_df)

        #Ensure that X has expected number of features  
        assert X.shape[1]==31,module_logger.exception('Arrays X of shape [:,31] expected.')
        #Ensure that X and y have same number of rows  
        assert X.shape[0]==y.shape[0],module_logger.exception('Arrays X and y should have same number of rows.') 

        module_logger.info('drop columns + encoding done')
        module_logger.info('pipeline.fit_transform completed successfully')

    elif method=='inverse':
        module_logger.info('Starting pipeline.inverse_transform')

        # all_preprocess cannot be None
        assert all_preprocess!=None, module_logger.error('Missing pipeline object all_preprocess.')

        X=all_preprocess.inverse_transform(input_df)
        y=[] #return empty array as inverse transform only applies to input features
        module_logger.info('drop columns + encoding done')
        module_logger.info('pipeline.inverse_transform completed successfully')

    
    module_logger.info('Building features complete.')
    
    #save pipeline when method is fit, fit_transform
    if method in ['fit','fit_transform']:
        dump(build_feature_pipe,open('./models/build_feature_pipe.pkl','wb')) 
        dump(all_preprocess,open('./models/preprocess_pipe.pkl','wb')) 
        module_logger.info('Pipeline saved.')
    
    return X, y

if __name__ == '__main__':
    import make_dataset
    from pickle import load

    data_files_list_path='./data/interim/LCA_files_list.txt'

    input_df=make_dataset.main(path=data_files_list_path,file_type='file_list')

    #For training the very first time
    #X,y = main(input_df, build_feature_pipe=None, all_preprocess=None, method='fit')
    #print(X.shape)
    #print(y.shape)

    #For iterative training
    build_feature_pipe=load(open('./models/build_feature_pipe.pkl','rb')) 
    all_preprocess=load(open('./models/preprocess_pipe.pkl','rb')) 
    X,y = main(input_df,build_feature_pipe,all_preprocess, method='inverse')
    print(X.shape)
    print(y.shape)