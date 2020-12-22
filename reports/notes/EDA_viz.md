# Exploratory Data Analysis and related Visualizations
## Questions that can be asked and answered -  
Try answering this for PERM and LCA based programs.  
1. How has the outcome changed over the last 10 years?  
Stacked bar chart comparing count for each outcome from 2010 - 2020.  
2. Are some employers, Soc_Codes, employer locations more prone to be denied vs approved?  
a. employers, Soc_codes - List of top 10 employers, Soc_codes that have highest denials  
b. location - plot standardized denial rate on map per employer_state, per year. Show values on hover.  
3. How has the geographical distribution of different programs changed over time?  
plot number of applications on map per employer_state, per program, per year. show case status distribution on hover.  
4. How has the distribution of different programs changed over Soc_codes?  
Keeping current year as baseline, show variance values (red if less, green if more) for each year (column) and soc_code (row)
5. How is the average distribution of new applications received per month (Received_date) compared with applications decided that month?  
Candlelight charts? Heat maps?  
6. How has the processing days (Decision_date - Received_date) changed over time (months and years) - month=Decision month?
Bar chart plotted over month/year scale  


## XAI -  
1. Candidate and outlier records  
2. Feature importance (after training the model)  
 