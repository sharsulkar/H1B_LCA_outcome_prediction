import numpy as np
np.random.seed(42)
import pandas as pd
import logging
from pickle import dump, load
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

module_logger= logging.getLogger('my_application.train_eval_predict')

def main(model=None,action='train',X=[],y=[]):
    """Method to train a model or evaluate the trained model.

    Args:
        model (sklearn model object, optional): The trained model. Required if action='evaluate'. Defaults to None.
        action (str, optional): Valid values are ['train','evaluate']. Defaults to 'train'.
        X (list, optional): Preprocessed training or evaluation data. Defaults to [].
        y (list, optional): Target variable. Defaults to [].
    """
    if model is None:
        model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=100),n_estimators=200,learning_rate=0.1,random_state=10)
    else:
        model=model
    
    if action=='train':
        print(X.shape)
        print(y.shape)
        #assert X and y cannot be empty
        assert len(X)!=0, module_logger.error('X cannot be empty when training model')
        assert len(y)!=0, module_logger.error('y cannot be empty when training model')

        model.fit(X,y)
        module_logger.info('Model fit complete.')

        #save the model
        dump(model,open('./models/adaboost_batch_train.pkl','wb'))
        module_logger.info('Model saved.')

    elif action=='evaluate':
        #assert X and y cannot be empty
        assert len(X)!=0, module_logger.error('X cannot be empty when evaluating model')
        assert len(y)!=0, module_logger.error('y cannot be empty when evaluating model')
        
        module_logger.info('Model evaluation complete.')
        f1=f1_score(y_true=y,y_pred=model.predict(X),average=None)
        print('f1-score:',f1)
        module_logger.info('Model evaluation metrics: [%d,%d]',f1[0],f1[1])


if __name__ == '__main__':
    import make_dataset
    import build_features
    from pickle import load

    #data_files_list_path='./data/interim/LCA_files_list.txt'
    #input_df=make_dataset.main(path=data_files_list_path,file_type='file_list')

    #For training the very first time
    #'''
    data_path='./data/interim/LCA_dataset_sample10000.xlsx' 
    input_df=make_dataset.main(path=data_path,file_type='data_file')
    X,y = build_features.main(input_df, build_feature_pipe=None, all_preprocess=None, method='fit_transform')
    main(model=None,action='train',X=X,y=y)
    #'''

    #for incremental training of existing model
    '''
    data_path='./data/interim/LCA_dataset_sample1000.xlsx' 
    input_df=make_dataset.main(path=data_path,file_type='data_file')
    X,y = build_features.main(input_df, build_feature_pipe=None, all_preprocess=None, method='fit_transform')
    model=load(open('./models/adaboost_batch_train.pkl','rb'))
    main(model=model,action='train',X=X,y=y)
    '''

    #for evaluating existing model
    '''
    data_path='./data/interim/LCA_dataset_sample1000.xlsx' 
    input_df=make_dataset.main(path=data_path,file_type='data_file')
    build_feature_pipe=load(open('./models/build_feature_pipe.pkl','rb')) 
    all_preprocess=load(open('./models/preprocess_pipe.pkl','rb')) 
    X,y = build_features.main(input_df, build_feature_pipe=build_feature_pipe, all_preprocess=all_preprocess, method='transform')
    model=load(open('./models/adaboost_batch_train.pkl','rb'))
    main(model=model,action='evaluate',X=X,y=y)
    '''





