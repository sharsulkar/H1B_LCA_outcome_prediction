Preprocessing pipeline to transform source data
------------------------------------------------

The source data undergoes multiple preprocessing steps before it can be used to train the ML model. The preprocessing steps are listed below -

**1. Drop rows -**

Source data has 4 target classes, we are only interested in the classes ['Confirmed','Denied']. 
The rows in the source data that do not belong to these two classes are dropped in this step. 
The resulting dataframe's index is also reset.
Please refer `DropRowsTransformer <https://sharsulkar.github.io/H1B_LCA_outcome_prediction\transforms.html#src.transforms.DropRowsTransformer>`_ in the next 
section for more details on implementation of this step.

**2. Feature engineering -**

Based on the exploratory data analysis observations and suggestion, 7 new features are generated using the existing source features. 
The list of existing `source features  <https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/feature_engineering_columns.csv>`_ 
used for building new features is available on github. 
Also please refer `BuildFeaturesTransformer <https://sharsulkar.github.io/H1B_LCA_outcome_prediction/transforms.html#src.transforms.BuildFeaturesTransformer>`_ 
in the next section for more details on implementation of this step.

**3. Drop features that are not important for training -**

After the feature engineering step, the redundant features are dropped in this step. 
The list of `features <https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/drop_columns.csv>`_ that are dropped is 
available on github. 
Also please refer `DropFeaturesTransformer <https://sharsulkar.github.io/H1B_LCA_outcome_prediction/transforms.html#src.transforms.DropFeaturesTransformer>`_ 
in the next section for more details on implementation of this step.

**4. Encode categorical features -**
A custom encoder for categorical features is specially built for this application. The out-of-the-box encoders available in sklearn like label_encoder, 
onehotencoder etc are not sufficient to encode real world data. This custom encoder is better suited for this application because -

a. ML models converge much faster during training when the data is centered and scaled. So random standard normal distribution is used to encode categorical values.

b. The custom encoder stores the categories and their encodings as key-value pairs. This is very helpful when training in batches as the encoding from previous 
batches is not replaced and key-value integrity can be enforced to ensure same numeric encoding is not applied to two different keys.

The list of `categorical features <https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/categorical_columns.csv>`_ is 
available on github repo. 
Also please refer `RandomStandardEncoderTransformer <https://sharsulkar.github.io/H1B_LCA_outcome_prediction/transforms.html#src.transforms.RandomStandardEncoderTransformer>`_ 
in the next section for more details on implementation of this step.

**5. Impute and scale numeric features -**

The numeric features are first imputed with their median values. Then the numeric features are scaled using a custom built version sklearn's StandardScaler. 
Custom scaler is needed because the sklearn version computes sample statistics on the training data which is not useful for batch training. The statistics computed for previous batches should not be replaced by future batches. Instead of sample statistics, pooled statistics are used so batch training will incrementally update the mean and variance parameters of the custom scaler.

The list of `numeric features <https://raw.githubusercontent.com/sharsulkar/H1B_LCA_outcome_prediction/main/data/processed/numeric_columns.csv>`_ is available on 
github repo. 
Also please refer `CustomStandardScaler <https://sharsulkar.github.io/H1B_LCA_outcome_prediction/transforms.html#src.transforms.CustomStandardScaler>`_ in 
the next section for more details on implementation of this step.

.. automodule:: src.build_features
   :members: