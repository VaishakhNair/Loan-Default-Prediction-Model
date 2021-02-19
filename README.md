(P.S Github sometimes takes time while loading ipynb files, so just reload a couple times if it shows you error. or you can render it using https://nbviewer.jupyter.org/. Open the link and just copy paste the link of ipynb file. It should display just fine then :) )

# Loan-Default-Prediction-Model

A binary prediction model for loan defaults.

PROBLEM : Build a binary prediction model that allows us to predict the likelihood of default of a consumer.

APPROACH : After observing the dataset initially, it was understood that the target variable is "loan_default". The target variable has two values '0' or '1'. This helped understand that the above problem can be solved by Supervised Learning Classification Approach.


Roadmap followed for the problem :
1) EXPLORATORY DATA ANALYSIS : Analysing the dataset and trying to understand the values, what they represent etc.
2) FEATURE ENGINEERING : This stage involves converting the values in the dataset into the type which can be easily interpreted by the ML model.
3) FEATURE SELECTION : Deciding on which features should be fed to the model and which should be dropped.
4) MODEL CREATION : Since this is a supervised classification problem the following algorithms will be applied - Decision Tree, Naive Bayes, Random Forest, Logistic Regression, K- nearest neighbour. Hyperparameter Optimization was applied to each one of them inorder to obtain an accurate model.



Brief Description of EDA :
1) Observed the columns, shape, values in the dataset.
2) Observed the number of unique values in each feature.
3) Observed the datatypes of the features.
4) The "3.286%" of missing categorical values in Employment.Type was handled by utilizing the mode value of the column.
5) No duplicate values found.

Analysed :
1) The value count and histogram of unique values in Aadhar_flag, PAN_flag, VoterID_flag, Driving_flag, Passport_flag, Employment.Type.
2) The value distribution of Loan default
3) Histogram, Distribution Plot, Transformation and Transformed plot for Disbursed Amount, asset_cost, ltv, PERFORM_CNS.SCORE, PRIMARY.INSTAL.AMT, SEC.INSTAL.AMT. Used log tranforms for all values except ltv as resultant skew was high. used boxcox for ltv
4) How Disbursed Amount, asset_cost, ltv values are distributed wrt Employment type by plotting grouped boxplots
5) Histograms for State_ID, manufacturer_id, branch_id, supplier_id, NO.OF_INQUIRIES, Current_pincode_ID, Employee_code_ID, DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS, NEW.ACCTS.IN.LAST.SIX.MONTHS
6) The cross tabulation between [ State_ID, manufacturer_id, branch_id, supplier_id, NO.OF_INQUIRIES, Current_pincode_ID, Employee_code_ID, DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS, NEW.ACCTS.IN.LAST.SIX.MONTHS ] with "loan_default" to understand the how the values are related to the default of loans.
7) Unique values in PERFORM_CNS.SCORE.DESCRIPTION, plotted the categorical plot for unique values with disbursed_amount, asset_cost, ltv, PRIMARY.INSTAL.AMT, SEC.INSTAL.AMT
8) Distributions of PRI.NO.OF.ACCTS, PRI.ACTIVE.ACCTS, PRI.OVERDUE.ACCTS, PRI.CURRENT.BALANCE, PRI.SANCTIONED.AMOUNT, PRI.DISBURSED.AMOUNT
9) Distributions of SEC.NO.OF.ACCTS, SEC.ACTIVE.ACCTS, SEC.OVERDUE.ACCTS, SEC.CURRENT.BALANCE, SEC.SANCTIONED.AMOUNT, SEC.DISBURSED.AMOUNT
10) Correlation heatmap, isolated values above 0.7.

Brief description of feature engineering:
1) Observed the data types and found 6 features with "Object" datatypes which must be converted to numerical values before giving the data to ML model.
2) Converted the Date.of.Birth feature with object datatype into AGE with numerical value and plotted the histogram for the same for ease of analysis
3) Label Encoded Employment.Type (Two unique values inside the feature was converted into "0" and "1" respectively
4) Reduced the unique values PERFORM_CNS.SCORE.DESCRIPTION by Combining and renaming Various unscored data into one umbrella of "Not scored"
5) Label encoded and One hot encoded the PERFORM_CNS.SCORE.DESCRIPTION
6) Converted the DisbursalDate with object data type into a feature with numerical values signifying Number of days since disbursal and plotted the histogram of the ame for ease of analysis
7) Converted AVERAGE.ACCT.AGE and CREDIT.HISTORY.LENGTH with Object datatype into Number of months. Plotted the histogram for the same.

Brief description for feature selection :
1) UniqueID had all unique values, MobileNo_Avl_Flag had only a single value, both were dropped.
2) Removed the highly correlated values (> 0.7) to remove redundancy.
3) Removed other additional leftover columns created due to feature engineering
4) Final data set had the following dimensions (233154, 46)


Brief Description of Model Creation
1) Performed scaling on the data
2) Performed PCA for reduction in dimensionality ( 46 to 10 ) to improve the process speed
3) Train test split of the data into (80/20)
4) Implemented DT, NB, RF, LR, KNN and performed hyperparameter optimization for each algorithm using GridsearchCV / RandomSearchCV
