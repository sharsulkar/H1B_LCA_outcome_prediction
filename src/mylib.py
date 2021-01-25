import pandas as pd

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
    return list(pd.read_csv(filepath, header=None, squeeze=True))

def modify_observations(df,index,columns,values,modify_action='update_values'):
  #assert - index, columns and values are string list type, 
  #columns and values are same size, for single column - value should be scalar
  #columns that have modification exist in observation_df
  #
  if modify_action=='add_row':
    df.loc[index]=values

  elif modify_action=='update_values':
    df.loc[index,columns]=values
  
  return df

  def missing_statistics(df,column):
    return (df.shape[0]-df[column].count())*100/df.shape[0]

  def cardinality_statistics(df,column):
    return (df.shape[0]-len(df[column].unique()))*100/df.shape[0]
    