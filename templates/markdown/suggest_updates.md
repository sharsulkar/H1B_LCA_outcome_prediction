## SUGGEST CHANGES TO USER APPLICATION TO CHANGE OUTCOME FROM DENIED TO CONFIRMED  
As an applicant to OFLC's LCA programs, a denial prediction can be heart breaking. Keeping aside the slight chance of this being a false prediction, there is nothing more helpful to the applicant that knowing a way to turn the prediction around to *Confirmed*. This tool does just that, for all denial predictions -  
* the tool will run through all possible combinations of valid values for *select fields* in the application and check if any of the combination results in a prediction to *Confirmed*.  
* If more than one combination has a positive prediction, the tool will return the combination with least possible changes to the application.  

### APPROACH TO GENERATING SUGGESTIONS  
There can be many approaches to solve this problem. For instance, one could use clustering algorithms to find the closest training data that has been confirmed and suggest its features to the user. Others might derive the decision boundaries of the model and use that to suggest a baseline levels for each feature values that will result in either outcome.
While all these are valid approaches, the easiest of them all is to create a grid of all possible valid values of the features and have our trained model predict each row in the grid. Assuming the model has good performance on previously unseen data, this approach yeilds the possible combinations that have a chance of changing the outcome with the least amount of effort as well as memory and processing overhead. 
The details of how this works has been described below -  

**Step 0 - Enter application details and make prediction** -   
The process is triggered when the user enters the application details. The tool runs the application parameters through the preprocess pipeline described earlier and makes a prediction. The prediction can either be a confirmed application or a denied application. If the application is expected to be confirmed, there are no further steps to process and the tool stops there.
If the application is expected to be denied, below steps are executed in their given order.  

**Step 1 - For denied application, generate a grid of possible valid values for *select features*** -  
The model is trained on and makes predictions based on values in 31 features listed here <enter link>. Out of these, the applicant has some degree of control over what values are entered only on a subset of these 31 features. For instance, the applicant has no control over the Employer Name, SOC Code and title or NAICS code. On the other hand, the application can negotiate changing the wage level, the duration of LCA validity or in some cases, the worksite location.  
Given this, the grid of valid values generated is only for the features the applicant has some control over. The list of features and the valid values considered are as given below -  

FULL_TIME_POSITION - ['Y','N']  
PW_WAGE_LEVEL - ['I','II','III','IV']  
AGREE_TO_LC_STATEMENT - ['Y','N']  
H-1B_DEPENDENT - ['Y','N']  
WILLFUL_VIOLATOR - ['Y','N']  
PUBLIC_DISCLOSURE - ['Disclose Business', 'Disclose Employment','Disclose Business and Employment']  
AGENT_REPRESENTING_EMPLOYER -  ['Y','N']   
EMPLOYER_WORKSITE_YN - ['Y','N']  
OES_YN - ['Y','N']  
WAGE_ABOVE_PW_HR - [0, *_value*, 10] where *_value*>0 is the amount given in the application  
VALIDITY_DAYS - [100, *_value*, 1095]  where *_value* \<1095 is the value given in the application  

The algorithm to create a grid of all possible combinations along with an example of its application to generating a 4-bit binary truth table is as given below -  
```
Let arr be the array of valid values for the selected features
For generating a truth table for 4-bit binary, arr=[[0,1],[0,1],[0,1],[0,1]]

1. Calculate an array of cumulative product of the length of each element in arr
cum_prod = cumulative_product(len(arr[i]) for i in range(len(arr)))
for our example, cum_prod=[2,4,8,16]

2. Calculate the absolute product of the length of all element in arr
m=product(len(arr[i]) for i in range(len(arr)))
for our example, m=16

3. Initialize the grid array with all possible combinations for the first feature. Each valid value in arr[0] will have to be repeated m/cum_prod[0] times. 
grid_arr=arr[0].repeat(m/cum_prod[0])
For our example, the first row will then be each value in [0,1] repeated m/cum_prod[0] = 8 times, which is
[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]

4. For each element in arr from index i= 1 to last, perform below steps:
    4.1 Let temp_arr be the temporary array to store expanded grid of values for features 
    temp_arr=arr[i].repeat(m/cum_prod[i])
    for our example, 
    when i=1, temp_arr=[0 0 0 0 1 1 1 1]
    when i=2, temp_arr=[0 0 1 1]
    when i=3, temp_arr=[0 1]
    
    4.2 To generate the grid correctly, we want to repeat and hstack temp_arr cum_prod[i-1])-1 number of times.
    For j in range(int(cum_prod[i-1])-1)):
        temp_arr=hstack(temp_arr,arr[i]).repeat(m/cum_prod[i]))
    for example, when i=1, and temp_arr=[0 0 0 0 1 1 1 1], we want to repeat it cum_prod[0]-1 = 1 times and hstack it to temp_arr which gives [0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1]
    similarly, when i=2, and temp_arr=[0 0 1 1], we want to repeat it cum_prod[1]-1=3 times and hstack it to temp_arr which gives [0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1], and so on.
    
    4.3 Vstack the temp_arr from previous step to grid_arr
    grid_arr=vstach(grid_arr,temp_arr)
    for example, when i=1, grid_arr=[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1] and temp_arr=[0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1],
    The grid_arr after this step will be [[0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1],[0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1]].
    Similarly for rest of the iterations.
    
5. Return the Transposed grid_arr so that number of columns match the number of features and each row is a unique combination of all the features.
return grid_arr.T
For our example, the returned array will be of shape (16,4) with the first few rows looking as below -
[[0 0 0 0]
 [0 0 0 1]
 :
 [1 1 1 1]]
End
```

**Step 2 - Generate a full sized array of all features using the grid generated in previous step** -  
The grid of all possible combinations only generates data for 11 select features. The remaining 20 features that will remain unchanged will be added to the grid in this step.  
To do that, follow the below steps -
```
Let X_denied be a dataframe that contains the original values of 31 features that was used to generate the denied prediction, let grid_arr be the (m,11) shaped array, that holds all possible combinations of the 11 selected features, where m = absolute product of the length of all element in arr, let var_arr be the array that stores the column names for the 11 selected features.
Then,
1. Create a new dataframe where X_denied is repeated m times and index are reset inplace -
X_reconstructed=X_denied.iloc[np.arange(1).repeat(grid_arr.shape[0])].reset_index(drop=True)

2. For each row in X_reconstructed, update only the columns in var_arr with values from grid_arr-
pd.DataFrame.update(X_reconstructed,pd.DataFrame(grid_arr,columns=var_columns))

3. Run X_reconstructed through the pipeline to get an array X_reconstructed_arr ready that can be fed to the model.
X_reconstructed_arr=preprocess_pipe.transform(X_reconstructed)
```

**Step 3 - Generate suggestions to change prediction from Denied to Confirmed** -  
As noted before, this approach uses the current model to predict outcomes on a grid of all possible values for select features. If a positive prediction is found for one or more rows in the grid, the next task is to find the row that results in the least possible change for the user. For this, we use the weighted Eucledian distance measure to select the best row. 
The weights are hard coded to give preference to changes in some features that are more easier to change than others. 
In this implementation, out of the 11 selected features listed above, the features that have higher priority and thus larger weight are PW_WAGE_LEVEL, VALIDITY_DAYS and WAGE_ABOVE_PW_HR.  
The weighted Eucledian distance is computed as below -
```
for two 1-d arrays a,b and weights array of same shape
Weighted Eucledian distance = sqrt((w*q*q).sum())
where q = a - b
```
The complete algorithm to make suggestion is given below -
```
Let y be the predictions of the model for all rows in X_reconstructed_arr where y=0 is confirmed, y=1 is denied, let distance_arr be the array to store the Eucledian distances between two rows.
1. if len(y[y==0])>0: (Check if the model predicted any row with outcome as Confirmed.)
    1.1 For each row in X_reconstructed_arr:
        1.1.1 if y==0 (prediction is 'confirmed'):
            1.1.1.1 Find the weighted Eucledian distance between the current row in X_reconstructed_arr and the original denied row stored in X_arr.if the distance==0, it means that the current_row in X_reconstructed_arr is the same as the original denied row, in that case, add a offset to the distance to prevent it from being the row with smallest distance. 
            1.1.1.2 Append the distance to distance_arr
    
    1.2 get the index for minimum Eucledian distance and store it in min_change_index
    min_change_index=argmin(distance_arr)
    
    1.3 For each column in X_arr that is different from the corresponding column in X_reconstructed_arr[min_change_index]:
        1.3.1 Let current_value be X_arr[column], let new_value be X_reconstructed_arr[min_change_index][column]
            1.3.1.1 if current_value is NaN, current_value = 'missing'
            1.3.1.2 if column name is VALIDITY_DAYS and current_value >=1094, do not make a suggestion to reduce the validity period, continue
            1.3.1.3 if column name is WAGE_ABOVE_PW_HR and current_value>new_value, do not make suggestion to reduce the wage, continue
            1.3.1.4 else suggest changing the current_value to new_value.
            
2. if len(y[y==0])=0: Notify the user that no possible combination resulted in a prediction of confirmed application. The model is not useful in making suggestions in this case.
```