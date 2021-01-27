import pandas as pd
import logging

# create logger
module_logger = logging.getLogger('my_application.mylib')

def read_csv_to_list(filepath, header=None, squeeze=True):
    """
        Read a CSV file into a list.

        Args:
            filepath (str): CSV file path
            header (int, list of int, optional): Row number(s) to use as the column names, and the start of the data. Defaults to None.
            squeeze (bool, optional): If the parsed data only contains one column then return a Series. Defaults to True.

        Returns:
            list: list of values from CSV file
        """
    module_logger.info('Executing read_csv_to_list')
    return list(pd.read_csv(filepath, header, squeeze))

def modify_observations(df,index,columns,values,modify_action='update_values'):
    """
    Function to modify a dataframe by inserting new index values or updating existing records. 
    This function only supports string labels as indexes.

    Args:
        df (DataFrame object): DataFrame to be modified
        index (list of str): The list of index labels to be modified
        columns (list of str): The list of column labels to be modified
        values (list): The list of values to be updated/inserted. The length of values must match the lenght of columns.
        modify_action (str, optional): Modification action to be perfomed. Choose between 'add_rows' and 'update_values'. Defaults to 'update_values'.

    Returns:
        DataFrame: Returns the modified DataFrame
    """
    module_logger.info('Starting executiion of modify_observations module.')

    #columns and values are of same length
    assert len(columns)==len(values),module_logger.error('Input given in columns and values must have equal length.')
    #Input indexes exist in df
    assert set(index).issubset(set(df.index)),module_logger.error('Index not found in given input DataFrame.')
    #Input columns exist in df
    assert set(columns).issubset(set(df.columns.values)),module_logger.error('Columns not found in given input DataFrame.')

    if modify_action=='add_row':
        df.loc[index]=values

    elif modify_action=='update_values':
        df.loc[index,columns]=values

    module_logger.info('Executiion of modify_observations module complete.')

    return df

def missing_statistics(df,column):
    """
    Calculate the percent missing values for the given column in the given DataFrame

    Args:
        df (DataFrame object): DataFrame on which the statistics will be calculated
        column (str): Column in the DataFrame

    Returns:
        float: percent missing records in the column
    """
    module_logger.info('Starting executiion of missing_statistics module.')

    #Input column exist in df
    assert set([column]).issubset(set(df.columns.values)),module_logger.error('Column not found in given input DataFrame.')

    module_logger.info('Executiion of missing_statistics module complete.')

    return (df.shape[0]-df[column].count())*100/df.shape[0]

def cardinality_statistics(df,column):
    """Calculate the cardinality (measure of uniqueness) for the given column in the given DataFrame

    Args:
        df (DataFrame object): DataFrame on which the statistics will be calculated
        column (str): Column in the DataFrame

    Returns:
        float: cardinality of the column
    """
    module_logger.info('Starting executiion of cardinality_statistics module.')

    #Input column exist in df
    assert set([column]).issubset(set(df.columns.values)),module_logger.error('Column not found in given input DataFrame.')

    module_logger.info('Executiion of cardinality_statistics module complete.')

    return (df.shape[0]-len(df[column].unique()))*100/df.shape[0]
    
