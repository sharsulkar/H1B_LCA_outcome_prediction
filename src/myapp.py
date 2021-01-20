import logging

def main():
    logging.basicConfig(filename='./myapp.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('Started')
    #make_dataset
    logging.info('Dataset imported')
    #build_features
    logging.info('Features built')
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