import pandas as pd
import logging
import read_csv_to_list

def main():
    required_features=['a','b']
    required_features=read_csv_to_list.read_csv_to_list('https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/required_features.csv',header=None,squeeze=True)
    try :
        input_df=pd.read_excel('https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx',usecols=required_features)
    except ValueError:
        logging.exception('ValueError: columns %s not found',required_features)
    return input_df

if __name__ == '__main__':
    main()