## SUGGEST CHANGES TO USER APPLICATION TO CHANGE OUTCOME FROM DENIED TO CONFIRMED
As an applicant to OFLC's LCA programs, a denial prediction can be heart breaking. Keeping aside the slight chance of this being a false prediction, there is nothing more helpful to the applicant that knowing a way to turn the prediction around to *Confirmed*. This tool does just that, for all denial predictions -  
* the tool will run through all possible combinations of valid values for *select fields* in the application and check if any of the combination results in a prediction to *Confirmed*.  
* If more than one combination has a positive prediction, the tool will return the combination with least possible changes to the application.  

The details of how this works has been described below -  
**Step 0 - Enter application details and make prediction** - 
The process is triggered when the user enters the application details. The tool runs the application parameters through the preprocess pipeline described earlier and makes a prediction. The prediction can either be a confirmed application or a denied application. If the application is expected to be confirmed, there are no further steps to process and the tool stops there.
If the application is expected to be denied, below steps are executed in their given order.  

**Step 1 - For denied application, generate a grid of possible valid values for *select features*** -
The model is trained on and makes predictions on a list of 31 features listed here <enter link>. Out of there, the applicant has some degree of control over what values are entered only on a subset of these 31 features. For instance, the applicant has no control over the Employer Name, SOC Code and title or NAICS code. On the other hand, the application can negotiate changing the wage level, the duration of LCA validity or in some cases, the worksite location.  
Given this, the grid of valid values generated is only for the features the applicant has some control over. The list of features and the valid values considered is given below -  

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