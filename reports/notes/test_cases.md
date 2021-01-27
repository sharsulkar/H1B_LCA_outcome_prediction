# Test case document for H1B LCA outcome prediction
---
## Test cases for mylib.py

### Function read_csv_to_list(filepath, header=None, squeeze=True)
1. Check whether file exists in given filepath
   Type - Sanity check
   Notification - Raise error: file not found
2. Check whether file format is CSV in given filepath
   Type - Sanity check
   Notification - Raise error: Expecting a CSV file

### Function modify_observations(df,index,columns,values,modify_action='update_values')
1. Ensure columns and values are of same length
    Type - Sanity check
    Notification - Raise error: columns and values must have equal length
2. Ensure given index is present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Index not found in DataFrame    
3. Ensure given columns are present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Columns not found in DataFrame 
4. Ensure df is a pandas DataFrame object
    Type - Sanity check
    Notification - Raise error: DataFrame object expected 
5. Ensure update_values, add_rows are the valid values for modify_action.
    Type - Valid value check
    Notification - Raise ValueError: Invalid value found in modify_action. Expecting on of ('update_values','add_rows')

### Function missing_statistics(df,column)
1. Ensure given column is present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Columns not found in DataFrame 
2. Ensure input DataFrame is empty
    Type - Sanity check
    Notification - Raise error: Cannot accept empty DataFrame, divide by zero exception will occur.
3. Ensure df is a pandas DataFrame object
    Type - Sanity check
    Notification - Raise error: DataFrame object expected 

### Function cardinality_statistics(df,column)
1. Ensure given column is present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Columns not found in DataFrame 
2. Ensure input DataFrame is empty
    Type - Sanity check
    Notification - Raise error: Cannot accept empty DataFrame, divide by zero exception will occur.
3. Ensure df is a pandas DataFrame object
    Type - Sanity check
    Notification - Raise error: DataFrame object expected 
---

## Test cases for transforms.py

### Class DropRowsTransformer(row_index, inplace, reset_index, X)
#### Method transform()
1. Ensure X is a pandas DataFrame object
    Type - Sanity check
    Notification - Raise error: DataFrame object expected 
2. Ensure given index is present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Index not found in DataFrame   
3. Ensure inplace, reset_index have True,False as valid values.
    Type - Valid value check
    Notification - Raise ValueError: Expecting a binary value

### Class DropFeaturesTransformer(columns, inplace, X)
#### Method transform()
1. Ensure X is a pandas DataFrame object
    Type - Sanity check
    Notification - Raise error: DataFrame object expected 
2. Ensure given columns are present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Column not found in DataFrame   
3. Ensure inplace have True,False as valid values.
    Type - Valid value check
    Notification - Raise ValueError: Expecting a binary value

### Class RandomStandardEncoderTransformer(cat_cols, categories, RSE, X)
#### Method fit(X)
1.Ensure given columns are present in the DataFrame
    Type - Sanity check
    Notification - Raise error: Column not found in DataFrame   

### Class BuildFeaturesTransformer(input_columns, X)
#### Method date_diff(date1, date2)
1. Ensure date1 and date2 are of type datetime
    Type - Valid value check
    Notification - Raise ValueError: Expecting a datetime value
2. Ensure that date1 and date2 are not null
    Type - Valid value check
    Notification - Raise ValueError: date1 and date2 cannot be null

#### Method is_usa(country)
1. Ensure country is of type str
    Type - Valid value check
    Notification - Raise ValueError: Expecting a str value

#### Method transform()
1. Ensure the columns required to build features are present in the input DataFrame.
    Type - Sanity check
    Notification - Raise error: Column not found in DataFrame   
2. Ensure the new features created by this method are present in the DataFrame that is being returned
    Type - Sanity check
    Notification - Raise error: Column not found in DataFrame 
---

## Test cases for myapp.py
### Output of make_dataset
1. Ensure that the input_df has required columns
    Type - Sanity check
    Notification - Raise error: Column not found in DataFrame 
2. Ensure input_df is not empty
    Type - Sanity check
    Notification - Raise error: DataFrame is empty
---

## Test cases for build_features.py
### After build_feature_pipe.fit_transform is applied
1. Ensure that there are no records with CASE_STATUS not in ['Certified','Denied'] after drop_rows
    Type - Sanity check
    Notification - Raise error: Unexpected values found in CASE_STATUS. Valid values should be in ['Certified','Denied'].

### After all_process.fit is applied
1. Ensure that X has expected number of features
    Type - Sanity check
    Notification - Raise error: Array of shape [:,31] expected.
2. Ensure that X and y have same number of rows
    Type - Sanity check
    Notification - Raise error: Arrays X and y should have same number of rows


