
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropRowsTransformer(BaseEstimator, TransformerMixin):
    """
    A class to drop rows from a DataFrame.

    Args:
        row_index (pandas index object) : A list of indexes that should be dropped from the DataFrame.
        inplace : x (default=True)
            
        reset_index : binary (default=True)
            Whether reindexing should be performed after drop action
    """

    def __init__(self, row_index, inplace, reset_index):
        """
        Constructs all the necessary attributes for the DropRowsTransformer object.

        row_index : pandas index object
            A list of indexes that should be dropped from the DataFrame.
        inplace : binary (default=True)
            Whether the action should be performed inplace or not
        reset_index : binary (default=True)
            Whether reindexing should be performed after drop action
        """
        self.row_index = row_index
        self.inplace=True
        self.reset_index=True

    def fit( self, X, y=None):
        """
        Fit the class on input dataframe

            Parameters:
                X (pandas DataFrame): input dataframe
                y : place holder, defaulted to None

            Returns:
                None
        """
        return self 
    
    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe

            Parameters:
                X (pandas DataFrame): input dataframe
                y : place holder, defaulted to None

            Returns:
                X : Transformed dataframe
        """
        X.drop(index=self.row_index,inplace=self.inplace)
        if self.reset_index:
            X.reset_index(inplace=True)
        return X

#Custom transformer to drop features for input feature list
class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
    """A class to drop features from a DataFrame.

    Args:
        columns (array or list): A list of columns that should be dropped from the DataFrame.
        inplace : (binary) : Whether the action should be performed inplace or not, defaulted to True.

    Returns:
        X : Transformed dataframe
    """
    def __init__(self, columns, inplace):
        """
        Constructs all the necessary attributes for the DropFeaturesTransformer object.

        Args:
            columns (array or list): A list of columns that should be dropped from the DataFrame.
            inplace : (binary) : Whether the action should be performed inplace or not, defaulted to True.
        """
        self.columns = columns # list of categorical columns in input Dataframe
        self.inplace=True

    def fit( self, X, y=None):
        """
        Fit the class on input dataframe

            Args:
                X (pandas DataFrame): input dataframe
                y : place holder, defaulted to None
        """
        return self 
    
    def transform(self, X, y=None):
        """
        Apply transforms on the input dataframe

            Args:
                X (pandas DataFrame): input dataframe
                y : place holder, defaulted to None

            Returns:
                X : Transformed dataframe
        """
        X.drop(columns=self.columns,inplace=self.inplace)
        return X

class RandomStandardEncoderTransformer(BaseEstimator, TransformerMixin):
    """[summary]

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]

    Returns:
        [type]: [description]
    """
    #Class Constructor
    def __init__( self, cat_cols, categories=None, RSE=None ):
        """[summary]

        Args:
            cat_cols ([type]): [description]
            categories ([type], optional): [description]. Defaults to None.
            RSE ([type], optional): [description]. Defaults to None.
        """
        self.cat_cols = cat_cols # list of categorical columns in input Dataframe
        self.categories = categories # Array of unique non-numeric values in each categorical column
        self.RSE = RSE # Array of Random Standard encoding for each row in categories
        
    def fit( self, X, y=None ):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        #Get a list of all unique categorical values for each column
        self.categories = [X[column].unique() for column in X[self.cat_cols]]
        #replace missing values and append missing value label to each column to handle missing values in test dataset that might not be empty in train dataset
        for i in range(len(self.categories)):
            if np.array(self.categories[i].astype(str)!=str(np.nan)).all():
            self.categories[i]=np.append(self.categories[i],np.nan)
        #compute RandomStandardEncoding 
        self.RSE=[np.random.normal(0,1,len(self.categories[i])) for i in range(len(self.cat_cols))]
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y=None):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        for i in range(len(self.cat_cols)):
            #X[str(self.cat_cols[i])].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
            X.loc[:,(str(self.cat_cols[i]))].replace(dict(zip(self.categories[i], self.RSE[i])),inplace=True)
        return X    