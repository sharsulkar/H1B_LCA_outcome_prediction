import pandas as pd
import logging
from mylib import read_csv_to_list
import logging

module_logger= logging.getLogger('my_application.make_dataset')
def main():

    module_logger.info('Starting execution of make_dataset module.')

    required_features=read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/required_features.csv',header=None,squeeze=True)
    try :
        input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx',usecols=required_features)
    except ValueError:
        module_logger.exception('ValueError: columns %s not found',required_features)
    
    module_logger.info('Execution of make_dataset module complete.')

    return input_df

if __name__ == '__main__':
    main()