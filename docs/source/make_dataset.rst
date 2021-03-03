Make training dataset from source data
-----------
The source data for this project is located at the `OFLC Performance page <https://www.dol.gov/agencies/eta/foreign-labor/performance>`_ .
We are using FY20 dataset for training which is stored at source in 4 separate files - one for each quarter for FY20.
The make_dataset module allows us to source these files in two ways -  

1. As a text file with a list of file paths to source data files. 
make_dataset iterates over each file and returns a single dataframe with all source data appended into it as the output.

2. As a data file path. make_dataset returns a dataframe that has data from that file.


.. automodule:: src.make_dataset
   :members: