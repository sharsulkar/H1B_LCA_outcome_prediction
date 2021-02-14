## BACKGROUND  
When immigrating to USA on a employer sponsored work visa under any of the Department of Homeland Security (DHS) programs for specialty occupation, foreign applicants need to undergo Department of Labor (DOL) Certification to ensure that the local workforce is not adversely impacted due to jobs offered to them. Labor Condition Application (LCA) certification by DOL is the first step to apply for a non-immigrant visas for foreign workers. Although majority of LCA applications are approved, there is no tool currently available to accurately predict the outcome beforehand. Having the ability to predict the outcome and if possible amend an application that might get denied is a game changer, not just for applicants, but also for immigration professionals and the employers they represent.   

## SOURCE DATA INTRODUCTION  
Office of Foreign Labor Certification (OFLC) disclosure data is publicly available here https://www.dol.gov/agencies/eta/foreign-labor/performance  
The LCA disclosure data for **FY2020 - Q1 to Q4** will be used to train the model, which is available in xlsx format here  
FY20 Q1 data - https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q1.xlsx  
FY20 Q2 data - https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q2.xlsx  
FY20 Q3 data - https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q3.xlsx  
FY20 Q4 data - https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2020_Q4.xlsx  
The record layout for FY2020 dataset explaining each data field is available here https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Record_Layout_FY2020.pdf  

The most recently published LCA disclosure data available for FY2021 - Q1 will be used to validate the model.  
https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2021_Q1.xlsx  

The LCA performance statistics disclosed by OFLC are available here https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Selected_Statistics_FY2020.pdf  

Some interesting facts about 2020 LCA application statistics -  
1. OFLC processed 577,334 applications in FY2020, out of which  
    a. 558,626 (96.7%) were Certified  
    b. 3983 (0.7%) were Denied  
    c. 14725 (2.5%) were Withdrawn  

## PROJECT OBJECTIVE
This project will try to model the underlying process of LCA certification and provide solutions to below mentioned aspects of this process -  
1. Predict the outcome of a given application under the OFLC LCA program which cover H-1B, E-3 Australian, H-1B1 Chile, and H-1B1 Singapore visa categories.  
2. If the predicted outcome is 'Denied', suggest what changes to the application parameters will most likely change the outcome to 'Confirmed'.  

This will be solved as a two-class classification problem where *Confirmed* and *Denied* are the two classes. Applications that are *Withdrawn* will be kept out of training, validation, and prediction as withdrawing an application is a user initiated action, which may or may not be related to OFLC's decision making process, thus out of scope of the problem. The decision to confirm or deny an application is part of the decision making process of OFLC, which this project is trying to model.  
The model selection is also limited to shallow machine learning models available in sklearn library. Deep Neural Networks are purposefully kept out of scope.

## CHALLENGES
1. The training data is **extremely skewed** with 99.3% of all training data for Confirmed class with only 0.7% data available for Denied class. Training a well performing model with such inter-class disparity is going to be a challenge. Oversampling, feature engineering and model selection that can derive a good classification boundary is key to overcome this.  
2. There are more than 560K training records and each row has 96 features. Batch training and good feature selection is used for effective utilization of time and resources during training.  


