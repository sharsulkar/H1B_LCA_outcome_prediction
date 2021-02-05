import pandas as pd
import logging
from mylib import read_csv_to_list

module_logger= logging.getLogger('my_application.make_dataset')
def main(path,file_type='file_list'):

    module_logger.info('Starting execution of make_dataset module.')
    
    #import the required features list
    required_features=read_csv_to_list('./data/processed/required_features.csv',header=None,squeeze=True)

    if file_type=='file_list':
        #create an empty dataframe to hold the final concatenated result
        input_df=pd.DataFrame(columns=required_features)

        #define the file object 
        file_itr=open(file=path,mode='r')

        #iterate through the file and append the data to input_df
        for file_path in file_itr:

            try :
                data_df=pd.read_excel(file_path,usecols=required_features)

                module_logger.info('Imported dataframe with shape [%d,%d]',data_df.shape[0],data_df.shape[1])

                input_df=input_df.append(data_df,ignore_index=True)
            except ValueError:
                module_logger.exception('ValueError: columns %s not found',required_features)
        
        file_itr.close
    
    elif file_type=='data_file':
        input_df=pd.read_excel(path,usecols=required_features)


    module_logger.info('Final input dataframe is of shape [%d,%d]',input_df.shape[0],input_df.shape[1])
    module_logger.info('Execution of make_dataset module complete.')
    
    return input_df

if __name__ == '__main__':
    data_files_list_path='./data/interim/LCA_files_list.txt'
    input_df=main(path=data_files_list_path,file_type='file_list')
    print(input_df.shape)