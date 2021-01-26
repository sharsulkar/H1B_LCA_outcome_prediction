## Test cases for mylib.py
### read_csv_to_list(filepath, header=None, squeeze=True)
1. Check whether file exists in given filepath
   Type - Sanity
   Notification - Raise error: file not found
2. Check whether file format is CSV in given filepath
   Type - Sanity
   Notification - Raise error: Expecting a CSV file

### modify_observations(df,index,columns,values,modify_action='update_values')
1. Check whether columns and values are of same length
    Type - Sanity
    Notification - Raise error: columns and values must have equal length
2. Check whether given index is present in the DataFrame
    Type - Sanity
    Notification - Raise error: Index not found in DataFrame    
3. Check whether given columns are present in the DataFrame
    Type - Sanity
    Notification - Raise error: Columns not found in DataFrame  

### missing_statistics(df,column)
1. Check whether given column is present in the DataFrame
    Type - Sanity
    Notification - Raise error: Columns not found in DataFrame 

### cardinality_statistics(df,column)
1. Check whether given column is present in the DataFrame
    Type - Sanity
    Notification - Raise error: Columns not found in DataFrame 