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