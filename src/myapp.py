import logging
import make_dataset
import build_features


def main():
    logging.basicConfig(filename='./myapp.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Started')
    #make_dataset
    input_df=make_dataset.main()
    logging.info('Dataset imported')
    #build_features
    X,y=build_features.main(input_df)
    logging.info('Features built, X shape %d, y shape %d',X.shape,y.shape)

    #if mode==Train: 
        #train_model
    logging.info('Model trained')
    #elif mode==Predict:
        #make_prediction
    logging.info('Prediction made')
    #explain_model_prediction    
    logging.info('Model explained')

if __name__ == '__main__':
    main()