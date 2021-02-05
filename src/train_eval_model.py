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

    if model is None:
        model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=100),n_estimators=200,learning_rate=0.1,random_state=10)
    else:
        model=model
    
    if action=='train':
        #assert X and y cannot be empty
        assert X==[], module_logger.error('X cannot be empty when training model')
        assert y==[], module_logger.error('y cannot be empty when training model')

        model.fit(X,y_true)
        module_logger.info('Model fit complete.')

        #save the model
        dump(model,open('./models/adaboost_batch_train.pkl','wb'))
        module_logger.info('Model saved.')

    elif action=='evaluate':
        #assert X and y cannot be empty
        assert X==[], module_logger.error('X cannot be empty when evaluating model')
        assert y==[], module_logger.error('y cannot be empty when evaluating model')
        
        module_logger.info('Model evaluation complete.')
        f1=f1_score(y_true=y,y_pred=model.predict(X),average=None)
        module_logger.info('Model evaluation metrics: [%d,%d]',f1[0],f1[1])


if __name__ == '__main__':
    import make_dataset
    import build_features
    from pickle import load

