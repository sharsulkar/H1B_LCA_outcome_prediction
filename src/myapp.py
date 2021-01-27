import logging
import make_dataset
import build_features


def main():
    # create logger with 'spam_application'
    logger = logging.getLogger('my_application')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('./myapp.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    

    logger.info('Execution started')
    #make_dataset
    input_df=make_dataset.main()
    logger.info('Dataset imported')
    #build_features
    X,y=build_features.main(input_df)
    logger.info('Features built, X shape %d, y shape %d',X.shape,y.shape)

    #if mode==Train: 
        #train_model
    logger.info('Model trained')
    #elif mode==Predict:
        #make_prediction
    logger.info('Prediction made')
    #explain_model_prediction    
    logger.info('Model explained')

if __name__ == '__main__':
    main()