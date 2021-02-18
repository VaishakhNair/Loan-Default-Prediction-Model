#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # EXPLORATORY DATA ANALYSIS

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_train = pd.read_csv("train.csv")


# In[4]:


df_train.info()


# In[5]:


df_train.shape


# In[6]:


df_train.head(10)


# In[7]:


df_train.describe()


# In[8]:


#checking datatypes
df_train.dtypes


# In[ ]:





# In[9]:


#checking unique values
df_train.nunique()


# In[ ]:





# In[10]:


#plotting missing values
sns.set(rc={'figure.figsize':(15,7)})
sns.heatmap(df_train.isnull(), yticklabels= False, cbar=False, cmap ='viridis')


# In[11]:


#number of missing values
df_train.isnull().sum()


# In[12]:


#percent of missing values
percent_missing = df_train["Employment.Type"].isnull().sum() * 100 / len(df_train)
print(round(percent_missing,3))


# In[13]:


#mode values to replace missing values
print(df_train['Employment.Type'].mode())


# In[14]:


#replacing missing values
df_train['Employment.Type'].fillna(str(df_train['Employment.Type'].mode().values[0]), inplace=True)
df_train.isnull().sum()


# In[ ]:





# In[15]:


#checking for duplicate values
df_train[df_train.duplicated()]


# In[ ]:





# In[16]:


df_train['Aadhar_flag'].value_counts()


# In[17]:


df_train['PAN_flag'].value_counts()


# In[18]:


df_train['VoterID_flag'].value_counts()


# In[19]:


df_train['Driving_flag'].value_counts()


# In[20]:


df_train['Passport_flag'].value_counts()


# In[21]:


df_train['Employment.Type'].value_counts()


# In[ ]:





# In[22]:


sns.set(rc={'figure.figsize':(14,18)})
fig, ax =plt.subplots(3,2)
sns.countplot(df_train["Aadhar_flag"], palette=["#ffa872", "#72b4ff"], ax=ax[0,0]).set(title='Aadhar Flag', xlabel= " ", ylabel=" ")
sns.countplot(df_train["PAN_flag"], palette=["#ffa872", "#72b4ff"], ax=ax[0,1]).set(title='PAN Flag', xlabel= " ", ylabel=" ")
sns.countplot(df_train["VoterID_flag"], palette=["#ffa872", "#72b4ff"], ax=ax[1,0]).set(title='VoterID Flag', xlabel= " ", ylabel=" ")
sns.countplot(df_train["Driving_flag"], palette=["#ffa872", "#72b4ff"], ax=ax[1,1]).set(title='Driving Flag', xlabel= " ", ylabel=" ")
sns.countplot(df_train["Passport_flag"], palette=["#ffa872", "#72b4ff"], ax=ax[2,0]).set(title='Passport Flag', xlabel= " ", ylabel=" ")
sns.countplot(df_train["Employment.Type"], palette=["#ffa872", "#72b4ff"], ax=ax[2,1]).set(title='Employment Type', xlabel= " ", ylabel=" ")


# In[ ]:





# In[23]:


sns.set(rc={'figure.figsize':(7,5)})
plt.title('Loan Default')
sns.countplot(df_train["loan_default"], palette=["#BF55EC", "#F4D03F"])


# In[ ]:





# In[24]:


plt.figure(figsize=(15,5))
plt.title('Disbursed Amount')
df_train['disbursed_amount'].hist(bins=50, color = "#E26A6A")


# In[25]:


sns.set(rc={'figure.figsize':(15,5)})
plt.title('Distribution of Disbursed amount')
sns.distplot(df_train["disbursed_amount"], color ="#e84118")


# In[26]:


#log transform
df_train['disbursed_amount'] = np.log(df_train['disbursed_amount'])
plt.title('Transformed plot for Disbursed Amount')
sns.distplot(df_train["disbursed_amount"], color = "#1dd1a1")


# In[ ]:





# In[27]:


plt.title('Asset Cost')
df_train['asset_cost'].hist(bins=50, color = "#E26A6A")


# In[28]:


plt.title("Distribution of Asset Cost")
sns.distplot(df_train["asset_cost"], color ="#E26A6A")


# In[29]:


#log tranform
df_train['asset_cost'] = np.log(df_train['asset_cost'])
plt.title('Transformed plot for Asset Cost')
sns.distplot(df_train["asset_cost"], color = "#1dd1a1")


# In[ ]:





# In[30]:


plt.title('LTV')
df_train['ltv'].hist(bins=50, color = "#E26A6A")


# In[31]:


plt.title('Loan to Value Distribution')
sns.distplot(df_train["ltv"], color ="#E26A6A")


# In[32]:


#boxcox tranform
from scipy import stats
df_train["ltv"] = stats.boxcox(df_train['ltv'])[0]
plt.title('Transformed plot of LTV')
sns.distplot(df_train["ltv"], color = "#1dd1a1")


# In[ ]:





# In[33]:


#grouped boxplots
df_train.boxplot(column='disbursed_amount', by = 'Employment.Type')


# In[34]:


df_train.boxplot(column='asset_cost', by = 'Employment.Type')


# In[35]:


df_train.boxplot(column='ltv', by = 'Employment.Type')


# In[ ]:





# In[36]:


plt.figure(figsize=(15,5))
df_train['PERFORM_CNS.SCORE'].hist(bins=50, color = "#E26A6A")


# In[37]:


plt.title('Distribution of CNS SCORE')
sns.distplot(df_train["PERFORM_CNS.SCORE"], color ="#E26A6A")


# In[38]:


#log transform
df_train['PERFORM_CNS.SCORE'] = np.log1p(df_train['PERFORM_CNS.SCORE'])
plt.title('Transformed plot for CNS SCORE')
sns.distplot(df_train["PERFORM_CNS.SCORE"], color ="#1dd1a1")


# In[ ]:





# In[39]:


sns.set(rc={'figure.figsize':(15,11)})
fig, ax =plt.subplots(2,1)
sns.distplot(df_train["PRIMARY.INSTAL.AMT"], color = "#f567bc" , kde_kws={'bw': 0.1}, ax=ax[0]).set(title = "PRIMARY AMOUNT")
sns.distplot(df_train["SEC.INSTAL.AMT"], color = "#f567bc" , kde_kws={'bw': 0.1}, ax=ax[1]).set(title = "SECONDARY AMOUNT")


# In[ ]:





# In[40]:


#log transform
df_train['PRIMARY.INSTAL.AMT'] = np.log1p(df_train['PRIMARY.INSTAL.AMT'])
df_train['SEC.INSTAL.AMT'] = np.log1p(df_train['SEC.INSTAL.AMT'])

sns.set(rc={'figure.figsize':(15,10)})
fig, ax =plt.subplots(2,1)
sns.distplot(df_train["PRIMARY.INSTAL.AMT"], color = "#1dd1a1" , kde_kws={'bw': 0.1}, ax=ax[0]).set(title = "Tranformed plot for Primary Amount")
sns.distplot(df_train["SEC.INSTAL.AMT"], color = "#1dd1a1" , kde_kws={'bw': 0.1}, ax=ax[1]).set(title = "Tranformed plot for Secondary AMOUNT")


# In[ ]:


#cross tabulation for comparison wrt loan default 


# In[41]:


comparison_1 = pd.crosstab(df_train['State_ID'],df_train['loan_default'],margins = True)
comparison_1


# In[42]:


sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(df_train["State_ID"], palette= "Dark2")


# In[43]:


comparison_2 = pd.crosstab(df_train['manufacturer_id'],df_train['loan_default'],margins = True)
comparison_2


# In[44]:


sns.set(rc={'figure.figsize':(8,5)})
sns.countplot(df_train["manufacturer_id"], palette= "Dark2")


# In[45]:


comparison_3 = pd.crosstab(df_train['branch_id'],df_train['loan_default'],margins = True)
comparison_3


# In[46]:


sns.set(rc={'figure.figsize':(17,5)})
sns.countplot(df_train["branch_id"], palette= "Dark2")
plt.xticks(rotation = 90)


# In[47]:


comparison_4 = pd.crosstab(df_train['supplier_id'],df_train['loan_default'],margins = True)
comparison_4


# In[48]:


df_train['supplier_id'].hist(bins=50, color = "#59ABE3")


# In[49]:


comparison_5 = pd.crosstab(df_train['NO.OF_INQUIRIES'],df_train['loan_default'],margins = True)
comparison_5


# In[50]:


sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(df_train["NO.OF_INQUIRIES"], palette= "Dark2")


# In[51]:


comparison_6 = pd.crosstab(df_train['Current_pincode_ID'],df_train['loan_default'],margins = True)
comparison_6


# In[52]:


df_train['Current_pincode_ID'].hist(bins=50, color = "#F9BF3B")


# In[53]:


comparison_7 = pd.crosstab(df_train['Employee_code_ID'],df_train['loan_default'],margins = True)
comparison_7


# In[54]:


df_train['Employee_code_ID'].hist(bins=50, color = "#badc58")


# In[55]:


comparison_8 = pd.crosstab(df_train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'],df_train['loan_default'],margins = True)
comparison_8


# In[56]:


sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(df_train["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"], palette= "Dark2")


# In[57]:


comparison_9 = pd.crosstab(df_train['NEW.ACCTS.IN.LAST.SIX.MONTHS'],df_train['loan_default'],margins = True)
comparison_9


# In[58]:


sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(df_train["NEW.ACCTS.IN.LAST.SIX.MONTHS"], palette= "Dark2")


# In[ ]:





# In[59]:


#unique values
df_train["PERFORM_CNS.SCORE.DESCRIPTION"].unique()


# In[ ]:


#categorical plots of PERFORM_CNS.SCORE.DESCRIPTION wrt various values


# In[60]:


sns.set(rc={'figure.figsize':(15,5)})
sns.catplot(x='PERFORM_CNS.SCORE.DESCRIPTION', y='disbursed_amount', data= df_train, jitter = "0.25", height=6, aspect=3).set_xticklabels(rotation=90)


# In[61]:


sns.catplot(x='PERFORM_CNS.SCORE.DESCRIPTION', y='asset_cost', data= df_train, jitter = "0.25", height=6, aspect=3).set_xticklabels(rotation=90)


# In[62]:


sns.catplot(x='PERFORM_CNS.SCORE.DESCRIPTION', y='ltv', data= df_train, jitter = "0.25", height=6, aspect=3).set_xticklabels(rotation=90)


# In[63]:


sns.catplot(x='PERFORM_CNS.SCORE.DESCRIPTION', y='PRIMARY.INSTAL.AMT', data= df_train, jitter = "0.25", height=6, aspect=3).set_xticklabels(rotation=90)


# In[64]:


sns.catplot(x='PERFORM_CNS.SCORE.DESCRIPTION', y='SEC.INSTAL.AMT', data= df_train, jitter = "0.25", height=6, aspect=3).set_xticklabels(rotation=90)


# In[ ]:





# In[65]:


sns.set(rc={'figure.figsize':(15,15)})
fig, ax =plt.subplots(3,2)
sns.distplot(df_train["PRI.NO.OF.ACCTS"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[0,0])
sns.distplot(df_train["PRI.ACTIVE.ACCTS"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[0,1])
sns.distplot(df_train["PRI.OVERDUE.ACCTS"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[1,0])
sns.distplot(df_train["PRI.CURRENT.BALANCE"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[1,1])
sns.distplot(df_train["PRI.SANCTIONED.AMOUNT"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[2,0])
sns.distplot(df_train["PRI.DISBURSED.AMOUNT"], color = "#EB974E" , kde_kws={'bw': 0.1}, ax=ax[2,1])


# In[ ]:





# In[66]:


sns.set(rc={'figure.figsize':(15,15)})
fig, ax =plt.subplots(3,2)
sns.distplot(df_train["SEC.NO.OF.ACCTS"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[0,0])
sns.distplot(df_train["SEC.ACTIVE.ACCTS"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[0,1])
sns.distplot(df_train["SEC.OVERDUE.ACCTS"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[1,0])
sns.distplot(df_train["SEC.CURRENT.BALANCE"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[1,1])
sns.distplot(df_train["SEC.SANCTIONED.AMOUNT"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[2,0])
sns.distplot(df_train["SEC.DISBURSED.AMOUNT"], color = "#af44bb" , kde_kws={'bw': 0.1}, ax=ax[2,1])


# In[ ]:


#correlation heatmap


# In[67]:


plt.figure(figsize=(15,11))
corr = df_train.corr()
sns.heatmap(corr, annot = False, cmap = "YlOrBr")
plt.show()


# In[68]:


#returns correlation values
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])> threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[69]:


correlated_features = correlation(df_train, 0.7)
len(set(correlated_features))


# In[70]:


correlated_features


# In[ ]:





# # FEATURE ENGINEERING

# In[71]:


df_train.dtypes


# In[72]:


#converting to age in years
from pandas import DataFrame
from datetime import date 
import datetime

age_list_train = df_train['Date.of.Birth'].tolist()
age_years_train = []

#conversion of  date into age
for i in range(0, len(df_train)):
    today = date.today()
    format_str = '%d-%m-%y' # The format
    datetime_obj = datetime.datetime.strptime(age_list_train[i], format_str)
    age = today.year - datetime_obj.year - ((today.month, today.day) < (datetime_obj.month, datetime_obj.day))
    age_years_train.append(age)
    i += 1
    
#correction of years below 1970 giving negative values
age_years_corrected_train = []
for j in age_years_train:
    if j > 0:
        age_years_corrected_train.append(j)
    if j < 0:
        j = 100+j
        age_years_corrected_train.append(j)
        
ages_train = DataFrame (age_years_corrected_train,columns=['AGE'])
df_train.insert(7, 'AGE', ages_train)



df_train


# In[73]:


plt.figure(figsize=(15,5))
df_train['AGE'].hist(bins=50, color = "#D980FA")


# In[74]:


sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(df_train["AGE"], palette= "Dark2")


# In[ ]:





# In[75]:


#label encoding
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df_train['Employment_Type'] = labelencoder.fit_transform(df_train['Employment.Type'])
df_train


# In[ ]:





# In[76]:


#converting to number of days
today = pd.Timestamp('now')
df_train['DisbursalDate'] = pd.to_datetime(df_train['DisbursalDate'], format='%d-%m-%y')

df_train['Days_since_disbursal'] = (today - df_train['DisbursalDate'])
df_train['Days_since_disbursal']= df_train['Days_since_disbursal'].astype(str)
df_train[['Days_since_disbursal','delta']] = df_train['Days_since_disbursal'].str.split("days",expand=True)
df_train['Days_since_disbursal']= df_train['Days_since_disbursal'].astype(str).astype(int)
df_train = df_train.drop(columns= ['delta'])
df_train


# In[120]:


df_train['Days_since_disbursal'].hist(bins=50, color = "#65C6BB")


# In[ ]:





# In[77]:


#reducing number of unique values
df_train['PERFORM_CNS.SCORE.DESCRIPTION'].replace(to_replace=['Not Scored: More than 50 active Accounts found', 
                                                         'Not Scored: No Activity seen on the customer (Inactive)',
                                                         'Not Scored: No Updates available in last 36 months',
                                                         'Not Enough Info available on the customer','Not Scored: Only a Guarantor',
                                                         'Not Scored: Sufficient History Not Available',
                                                         'Not Scored: Not Enough Info available on the customer'], value= 'Not Scored', inplace = True)


# In[78]:


#reduced unique values
df_train["PERFORM_CNS.SCORE.DESCRIPTION"].unique()


# In[79]:


#label and OneHotEncoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
df_train['PCSD_LE'] = labelencoder.fit_transform(df_train['PERFORM_CNS.SCORE.DESCRIPTION'])
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df_train[['PCSD_LE']]).toarray())
df_train = df_train.join(enc_df)
df_train


# In[ ]:


#converting both features into number of months respectively


# In[80]:


df = df_train["AVERAGE.ACCT.AGE"].str.split(expand=True)
df1 = df[0].str.split('yrs',expand=True)
df2 = df[1].str.split('mon',expand=True)
df1[0] = df1[0].astype(int)
df2[0] = df2[0].astype(int)
df["Avg_Acc_Age_Months"] = (df1[0]*12) + df2[0]
df_train["AVERAGE.ACCT.AGE"] = df["Avg_Acc_Age_Months"]


# In[81]:


plt.figure(figsize=(10,4))
df_train['AVERAGE.ACCT.AGE'].hist(bins=50, color = "#EF4836")


# In[82]:


df = df_train["CREDIT.HISTORY.LENGTH"].str.split(expand=True)
df1 = df[0].str.split('yrs',expand=True)
df2 = df[1].str.split('mon',expand=True)
df1[0] = df1[0].astype(int)
df2[0] = df2[0].astype(int)
df["Avg_Acc_Age_Months"] = (df1[0]*12) + df2[0]
df_train["CREDIT.HISTORY.LENGTH"] = df["Avg_Acc_Age_Months"]


# In[83]:


plt.figure(figsize=(10,4))
df_train['CREDIT.HISTORY.LENGTH'].hist(bins=50, color = "#E87E04")


# In[ ]:





# # FEATURE SELECTION

# In[84]:


df_train.dtypes


# In[85]:


df_train.drop(["UniqueID","MobileNo_Avl_Flag"], axis=1, inplace = True)


# In[86]:


df_train.drop(["Date.of.Birth","Employment.Type","DisbursalDate","PERFORM_CNS.SCORE.DESCRIPTION","PCSD_LE"], axis=1, inplace = True)


# In[87]:


df_train.drop(correlated_features, axis=1, inplace = True)


# In[88]:


df_train.shape


# In[89]:


df_train.dtypes


# In[ ]:





# # MODEL CREATION with Hyperparameter Optimization

# In[90]:


X = df_train.drop("loan_default", axis =1)
y = df_train["loan_default"]


# In[91]:


#Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# In[92]:


#PCA
from sklearn.decomposition import PCA

pca=PCA(n_components=10)
pca.fit(X)
X_pca = pca.transform(X)


# In[93]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)


# In[ ]:





# In[125]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# **1) DECISION TREE**

# In[92]:


parameters = {'max_depth' : (3,5,7,9,10,15,20,25)
              , 'criterion' : ('gini', 'entropy')
              , 'max_features' : ('auto', 'sqrt', 'log2')
              , 'min_samples_split' : (2,4,6)}

DT_grid  = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = parameters, cv = 5, verbose = True)
DT_grid.fit(X_train,y_train)


# In[93]:


#best hyperparameters
DT_grid.best_estimator_


# In[96]:


#building DT model with best hyperparameters
DecisionTree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=5, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')

DecisionTree.fit(X_train,y_train)


# In[97]:


y_pred_dt = DecisionTree.predict(X_test)


# In[121]:


#confusion matrix
print(confusion_matrix(y_test, y_pred_dt))


# In[122]:


print(classification_report(y_test, y_pred_dt))


# In[123]:


#accuracy score of DT
print(accuracy_score(y_test, y_pred_dt))
print(balanced_accuracy_score(y_test, y_pred_dt))


# In[131]:


#kfolds cross validated output
kf = StratifiedKFold(n_splits=10)
predicted_dt = cross_val_predict(DecisionTree, X_train, y_train, cv=kf)
print('Accuracy Score :',accuracy_score(y_train, predicted_dt))
print('Report : ')
print(classification_report(y_train, predicted_dt))


# In[ ]:





# **2) NAIVE BAYES**

# In[99]:


np.random.seed(999)
nb_classifier = GaussianNB()

parameters_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

Gauss_NB = GridSearchCV(estimator=nb_classifier, 
                     param_grid= parameters_NB, 
                     cv=15,
                     verbose=1, 
                     scoring='accuracy')
Gauss_NB.fit(X_train, y_train)


# In[100]:


#best hyperparameters
Gauss_NB.best_estimator_


# In[102]:


#building NB model with best hyperparameters
GaussNB = GaussianNB(priors=None, var_smoothing=1.0)
GaussNB.fit(X_train, y_train)


# In[103]:


y_pred_gNB = GaussNB.predict(X_test)


# In[127]:


#confusion matrix
print(confusion_matrix(y_test, y_pred_gNB))


# In[128]:


print(classification_report(y_test, y_pred_gNB))


# In[129]:


#accuracy score
print(accuracy_score(y_test, y_pred_gNB))
print(balanced_accuracy_score(y_test, y_pred_gNB))


# In[130]:


#cross validated output
skf = StratifiedKFold(n_splits=10)
predicted_nb = cross_val_predict(GaussNB, X_train, y_train, cv=skf)
print('Accuracy Score :',accuracy_score(y_train, predicted_nb))
print('Report : ')
print(classification_report(y_train, predicted_nb))


# In[ ]:





# **3) RANDOM FOREST**

# In[106]:


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [2,4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[107]:


rf_Model = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)


# In[108]:


rf_Grid.fit(X_train, y_train)


# In[109]:


#best hyperparameters
rf_Grid.best_estimator_


# In[107]:


#building RF model with best hyperparameters
RndmFrst = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=2, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

RndmFrst.fit(X_train, y_train)


# In[108]:


y_pred_rf = RndmFrst.predict(X_test)


# In[132]:


print(confusion_matrix(y_test, y_pred_rf))


# In[133]:


print(classification_report(y_test, y_pred_rf))


# In[134]:


print(accuracy_score(y_test, y_pred_rf))
print(balanced_accuracy_score(y_test, y_pred_rf))


# In[135]:


skf = StratifiedKFold(n_splits=10)
predicted_rf = cross_val_predict(RndmFrst, X_train, y_train, cv=skf)
print('Accuracy Score :',accuracy_score(y_train, predicted_rf))
print('Report : ')
print(classification_report(y_train, predicted_rf))


# In[ ]:





# **4) Logistic Regression**

# In[130]:


param_lr = [{'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
             'C' : np.logspace(-4, 4, 20),
             'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
             'max_iter' : [100, 1000,2500, 5000]}]


# In[131]:


logreg = LogisticRegression()
logreg_grid = GridSearchCV(logreg, param_grid = param_lr, cv = 2, verbose=True, n_jobs=-1)
logreg_grid.fit(X,y)


# In[132]:


#best hyperparameters
logreg_grid.best_estimator_


# In[139]:


#building LR model with best hyperparameters
LogReg = LogisticRegression(C=0.0001, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

LogReg.fit(X_train, y_train)


# In[140]:


y_pred_lr = LogReg.predict(X_test)


# In[141]:


print(confusion_matrix(y_test, y_pred_lr))


# In[142]:


print(classification_report(y_test, y_pred_lr))


# In[143]:


print(accuracy_score(y_test, y_pred_lr))
print(balanced_accuracy_score(y_test, y_pred_lr))


# In[158]:


skf = StratifiedKFold(n_splits=10)
predicted_lr = cross_val_predict(LogReg, X_train, y_train, cv=skf)
print('Accuracy Score :',accuracy_score(y_train, predicted_lr))
print('Report : ')
print(classification_report(y_train, predicted_lr))


# In[ ]:





# **5) KNN**

# In[ ]:


#I had implemented the hyperparameter tuning of knn for this dataset in another notebook.
#Hence, copy pasting the code here and directly applying the tuned hyperparameters to the model

"""

gs = {'n_neighbors' : np.arange(1,25)}
knn_gs = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_gs, gs, cv = 2)
knn_grid.fit(X_train, y_train)

GridSearchCV(cv=2, error_score=nan,estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=5, p=2, weights='uniform'),
iid='deprecated', n_jobs=None, param_grid={'n_neighbors': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24])},
pre_dispatch='2*n_jobs', refit=True, return_train_score=False, scoring=None, verbose=0)
             
             
knn_grid.best_estimator_
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=24, p=2,
                     weights='uniform')
                     
                     

                     
"""


# In[151]:


#building KNN model with best hyperparameters
KNN = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=24, p=2,
                     weights='uniform')


# In[152]:


KNN.fit(X_train, y_train)


# In[153]:


ypred_knn = KNN.predict(X_test)


# In[154]:


#confusion matrix
print(confusion_matrix(y_test, ypred_knn))


# In[155]:


print(classification_report(y_test, ypred_knn))


# In[156]:


print(accuracy_score(y_test, ypred_knn))
print(balanced_accuracy_score(y_test, ypred_knn))


# In[159]:


skf = StratifiedKFold(n_splits=10)
predicted_rf = cross_val_predict(KNN, X_train, y_train, cv=skf)
print('Accuracy Score :',accuracy_score(y_train, predicted_rf))
print('Report : ')
print(classification_report(y_train, predicted_rf))


# In[ ]:





# **ACCURACIES OF CREATED MODELS**
# 
# 1. DECISION TREE           - 78.2348 %
# 
# 2. NAIVE BAYES             - 78.4864 %
# 
# 3. RANDOM FOREST           - 78.5185 %
# 
# 4. LOGISTIC REGRESSION     - 78.5185 %
# 
# 4. KNN                     - 78.3491 %
# 
# 
# 
# 
# 
