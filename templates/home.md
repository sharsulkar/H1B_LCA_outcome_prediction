## Predict the outcome of your LCA application accurately in no time  
### Welcome!
Immigrating to a new country can be stressful, especially if it to the Unites States of America, on the highly sought after speciality occupation work visas like H1B. With ever changing regulations, it is important for the employers, immigration attorneys and most importantly, the applicants themselves to be able to predict the outcome of an application as accurately as possible before submitting it.

This tool tries to do just that. With the power of machine learning and publicly available historical performance data published by US Department of Labor, it predicts the outcome of your LCA application based on the parameters entered by the user. The prediction is also explained based on the historically available data. For negative predictions, suggestions are made to improve chances of approval, based on similar applications available in the historically available data. 

### Readme
Before using this tool, please read the below guidelines on what the tool can and cannot do.  
1. **Immigrating to the US on a work visa is a multi-step process. What exactly does this tool predict?**  
The scope of this tool is limited to predicting the outcome of Labor Condition Application (LCA) which is most often the first step in applying for non-immigrant visas class. LCA decisions are handled by the US Department of Labor while the H1B work permits are granted by USCIS which comes under the Department of Homeland Security (DHS). You can find more information on LCA [here](https://en.wikipedia.org/wiki/Labor_Condition_Application#:~:text=The%20Labor%20Condition%20Application%20(LCA,1B%20for%20workers%20from%20Australia)).  

2. **Why should I trust the tool's prediction?**  
The machine learning model used to make predictions has been trained on the historical data for 2020 applications publicly made available on [OFLC website](https://www.dol.gov/agencies/eta/foreign-labor/performance). Although no model can predict the future with 100% certainty, this machine learning model has a F1-score of 99% for Confirmed applications and 90% for Denied applications when tested on data previously unseen by the model. Which means that although there is some chance of mis-classifying the given application, it is considerably low. More information on the internal working of the tool can be found in the documentation section <insert link here>.

3. **The approval rate of LCA applications is 99% currently, why do I need this tool?**  
It is true that in 2020, 99% of all LCA applications were confirmed, which is good news for the applicants as the power of probability is on their side. Nevertheless, immigration professionals can use this tool to come up with constraints on application parameters such that any application that meets the parameters will most likely be approved. The tool also suggests parameters in the application that can be modified to possibly change the outcome for a negative prediction.  

4. **The LCA application has many more fields than used here?**  
The tool uses only the input parameters that have the best correlation with the outcome. This is called feature selection and is a commonly used method to prevent training machine learning models on redundant parameters. This ensures efficient use of memory and time during training and prediction.

5. **Should I be concerned about data privacy? How is the data provided in the user input form used?**  
The data provided in the form is not personally identifiable data. The fields used the form is a subset of the fields available in the publicly available OFLC data that the model is trained on. The data in the input form is used only for making the prediction and is not stored in any backend database. The codebase for the tool is freely available on [github](https://github.com/sharsulkar/H1B_LCA_outcome_prediction) for public access and review.

