## TRAINING, VALIDATION AND TESTING THE MODEL
### TRAINING APPROACH  
The training approach was influenced by the following factors -  
1. Decision to solve the problem as a two-class classification supervised learning problem.  
2. Extremely skewed training data - 99.3% data belongs to class 0 and just 0.7% belongs to class 1.  
3. 150K + records available for training which include categorical features with high cardinality like EMPLOYER_NAME and SOC_TITLE, making a batch training approach more desirable.  
4. Training data available on the OFLC website in form of four quarterly files, each >50 Mb in size.  

To remedy some of the challenges listed above, during initial training and model selection, below 3 approaches were taken to decide which model might be the best fit to solve this problem -  
1. Undersampling class 0 and oversampling class 1 to get a equal distribution in each training batch.  
2. Solving the problem as an anomaly detection problem where class 1 is considered an anomaly.  
3. Using tree based classifiers on the skewed training data.  

The outcome of these 3 approaches on FY2020 Q1 dataset is as shown in the table below. Based on this, it was decided to proceed with using Adaboost trees on larger skewed batches.  

#### PERFORMANCE EVALUATION METRICS  
F1-score is selected as the evaluation metric because for class 1, avoiding false positives and true negatives was as important as improving the true positives.  

### MODEL SELECTION  
The AdaboostClassifier(n_estimators=200,learning_rate=0.1) striked the right balance between model complexity and validation scores and thus was selected for training on the full dataset.  

### MODEL TESTING 
#### TEST DATA
FY2021 Q1 data will be used to evalute performance of the trained model. This data is real-world, previously unseen and has the same class distribution as the training dataset.  

#### EVALUATION RESULTS

#### FEATURE IMPORTANCE
There are many approaches to explaining the model, feature importance being one of them. In addition to model performance metrics, getting the feature importance that makes sense to professionals in the field is key to correctly model the underlying decision making process. There are many ways to compute feature importance, the method used here is by feature exclusion. Usually for feature exclusion, the model is trained by excluding some features of the training data and comparing the prediction performance between various subsets to identify which excluded features impact the prediction performance the most. These features are then deemed to be most important.  
More easier way of doing this is to perform this with the evaluation dataset. To start with, the baseline performance metric is computed with all features. Then a feature mask is created to exclude one of the feature in the evaluation dataset and the model performance is calculated for this masked data. The mask is then moved to exclude each feature once, computing the evaluation metric each time. The metrics are then compared to identify changes in the model performance with respect to baseline to find important features.  
The outcome of this exercise on the selected model is as given below -  
