#!/usr/bin/env python
# coding: utf-8

# In[2]:


# for data wrangling
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


from collections import Counter

# for statistics and metrics
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (accuracy_score, 
                            classification_report,
                            recall_score, precision_score, f1_score,
                            confusion_matrix)


# for algorithms
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# SHAP
import shap

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
pd. set_option("display.max_columns", None)

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Import/Read the file
df = pd.read_csv('C:/Users/abhin/OneDrive/Documents/MGP/RTA Dataset.csv')


# In[4]:


df


# Exploratory Data Analysis

# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


# checking for any nullvalues in the dataframe.
df.isna().sum()


# In[10]:


df.describe()


# In[11]:


df.describe().T


# In[12]:


df.describe(include=['O']).T


# In[13]:


df['Accident_severity'].value_counts()


# In[14]:


# checking the target
plt.figure(figsize=(14,7))
# barplot
ax1 = plt.subplot(1,2,1)
cp = sns.countplot(x=df["Accident_severity"])
ax1.set_xlabel(" ")
ax1.set_ylabel(" ")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine(top=True, right=True)
# pieplot
ax2 = plt.subplot(1,2,2)
plt.pie(df["Accident_severity"].value_counts(),
        labels=list(df["Accident_severity"].unique()),
        autopct='%1.2f%%',
        pctdistance=0.8,
        shadow=True,
        radius=1.3,
        textprops={'fontsize':14}
       )
ax2.set_xlabel(" ")
plt.xlabel('Composition of "Accident Severity"', fontsize=15, labelpad=20)
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[15]:


df.isna().sum()


# In[16]:


df['Service_year_of_vehicle'].value_counts()


# In[17]:


df['Defect_of_vehicle'].value_counts()


# In[18]:


# Extracting hour and minute from timestamp.

df['hour'] = pd.to_datetime(df['Time']).dt.hour
df['minute'] = pd.to_datetime(df['Time']).dt.minute


# In[19]:


df


# In[20]:


plt.figure(figsize=(15,70))
plotnumber = 1

for col in df.drop(['Lanes_or_Medians', 'Road_allignment', 'Pedestrian_movement','Time','hour','minute'], axis=1):
    if plotnumber <= df.shape[1]:
        ax1 = plt.subplot(16,2,plotnumber)
        sns.countplot(data=df, y=col, palette='Dark2')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(col.title(), fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
    plotnumber +=1
plt.tight_layout()


# In[22]:


# Checking the Pedestrian Movement column

plt.figure(figsize=(10,5))
sns.countplot(data=df, y='Pedestrian_movement', palette = 'Dark2')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Pedestrian Movement', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# In[24]:


plt.figure(figsize=(10,3))
sns.countplot(data=df, y='Lanes_or_Medians', palette = 'Dark2')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Lanes', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# In[26]:


plt.figure(figsize=(10,3))
sns.countplot(data=df, y='Road_allignment', palette = 'Dark2')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Road Allignment', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# In[27]:


plt.figure(figsize=(10,5))
sns.countplot(data=df, y='hour')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Hour', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# In[28]:


plt.figure(figsize=(10,15))
sns.countplot(data=df, y='minute')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Minute', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# Observations:
# 
# 1. Most of the accidents occured on Fridays, the least being sunday.
# 2. Age band of the drivers commiting the accidents around 18-30 years of age.
# 3. Sex of the accident commiting drivers Males mostly

# In[29]:


min = list(range(5,56, 5))
def convert_minutes(x: int):
    for m in min:
        if x % m == x and x > m-5:
            return m
        if x in [56,57,58,59]:
            return 0
        if x in min+[0]:
            return x


# In[30]:


df['minute'] = df['minute'].apply(lambda x: convert_minutes(x))


# In[31]:


plt.figure(figsize=(5,7))
sns.countplot(data=df, y='minute')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Minute', fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()


# In[34]:


num_cols = df.dtypes[df.dtypes == 'int64'].index.tolist()
cat_cols = df.dtypes[df.dtypes != 'int64'].index.tolist()
df[cat_cols]


# In[35]:


df[num_cols]


# In[36]:


# Dropping the time column

df.drop('Time', axis=1, inplace=True)


# In[37]:


df.isna().sum()[df.isna().sum() != 0]


# Handling Missing values through imputation

# In[38]:


impute_cols = [x for x in df.isna().sum()[df.isna().sum() != 0].index.tolist()]
for feat in impute_cols:
    mode = df[feat].mode()[0]
    df[feat].fillna(mode, inplace=True)


# In[40]:


df.isna().sum()


# Encoding

# In[41]:


def ordinal_encoder(df, feats): 
    for feat in feats:    
        feat_val = list(1+np.arange(df[feat].nunique()))
        feat_key = list(df[feat].sort_values().unique())
        feat_dict = dict(zip(feat_key, feat_val))
        df[feat] = df[feat].map(feat_dict)
    return df

df = ordinal_encoder(df, df.drop(['Accident_severity'], axis=1).columns)
df.shape


# In[42]:


df


# In[43]:


for col in df.drop('Accident_severity', axis=1):
    g = sns.FacetGrid(df, col='Accident_severity', size=6, aspect=1, sharey=False)
    g.map(sns.countplot, col, palette = 'Dark2')
    plt.show()


# Correlation

# In[44]:


plt.figure(figsize=(22,17))
sns.set(font_scale=0.8)
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.CMRmap_r)


# Upsampling

# In[46]:


X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[47]:


# upsampling using smote

counter = Counter(y_train)

print("=============================")

for k,v in counter.items():
    per = 100*v/len(y_train)
    print(f"Class= {k}, n={v} ({per:.2f}%)")

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

counter = Counter(y_train)

print("=============================")

for k,v in counter.items():
    per = 100*v/len(y_train)
    print(f"Class= {k}, n={v} ({per:.2f}%)")

print("=============================")

print("Upsampled data shape: ", X_train.shape, y_train.shape)


# In[48]:


y_test


# In[49]:


y_test = ordinal_encoder(pd.DataFrame(y_test, columns = ['Accident_severity']), pd.DataFrame(y_test, columns = ['Accident_severity']).columns)['Accident_severity']
y_train = ordinal_encoder(pd.DataFrame(y_train, columns = ['Accident_severity']), pd.DataFrame(y_train, columns = ['Accident_severity']).columns)['Accident_severity']


# In[61]:


get_ipython().system('pip install --upgrade scikit-learn numpy pandas catboost')


# In[74]:


get_ipython().system('pip install lightgbm')


# In[78]:



from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
#from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier


import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# In[81]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB



gbc = GradientBoostingClassifier(random_state = 0, learning_rate=0.45)
rfc = RandomForestClassifier(random_state = 0) 
lr = LogisticRegression(random_state = 0) 
dtc = DecisionTreeClassifier(random_state = 0) 
svc = SVC(random_state = 0) 
extree = ExtraTreesClassifier()
NBC2=GaussianNB()  
NBC1=CategoricalNB()
knn = KNeighborsClassifier(n_neighbors=10)
#CBC=CatBoostClassifier(random_state=42, n_estimators = 50)
#XGBR = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
clf = lgb.LGBMClassifier()
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=0)

lst = []

for i in(gbc,rfc,dtc,lr,svc,extree,NBC2,NBC1,knn,clf,ada):
    i.fit(X_train, y_train)

    i_pred = i.predict(X_test)

    cm = confusion_matrix(y_test, i_pred)

    cr = classification_report(y_test, i_pred)
    i_acc = round(accuracy_score(y_test, i_pred), 4)

    lst.append(i_acc)

    print(i,':\n','The confusion matrix:\n',cm,'\n')

    print('The classification report:\n',cr,'\n')

    print('-'*60)


# Checking the accuracy score of various models acsending order:

# In[82]:


Table = pd.DataFrame({'Model':['Gradient Boosting Classifier','Random Forest Classifier','Logistic Regression','Decision Tree Classifier','SVC','ExtraTreesClassifier','GaussianNB','CategoricalNB' ,'Knn', 'Lgbm','Ada'],
                     'Acc_Score': lst})

Table.sort_values('Acc_Score', ascending = False)


# Extrees provides the best accuracy out of all so we proceed with it

# In[83]:


extree.fit(X_train, y_train)
y_pred = extree.predict(X_test)
extree.get_params()


# Hyperparameter Tuning

# Trial 1

# In[85]:


gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)

# A parameter grid for XGBoost
params = {
    'n_estimators': range(100, 500, 100),
    'ccp_alpha': [0.0, 0.1],
    'criterion': ['gini'],
    'max_depth': [5,11],
    'min_samples_split': [2,3],
}

extree_estimator = ExtraTreesClassifier()

gsearch = GridSearchCV(
    estimator= extree_estimator,
    param_grid= params,
    scoring='f1_weighted',
    cv=gkf,
)

extree_model = gsearch.fit(X=X_train, y=y_train)
(gsearch.best_params_, gsearch.best_score_)


# Trial 2

# In[86]:


gkf2 = KFold(n_splits=3, shuffle=True, random_state=101).split(X=X_train, y=y_train)

params2 = {
    'n_estimators': range(300, 800, 100),
    'max_depth': [11,15],
    'min_samples_split': [2,3],
    'class_weight': ['balanced', None],

}

extree2 = ExtraTreesClassifier(ccp_alpha = 0.0,
                                criterion = 'gini',
                                max_depth = 11,
                                min_samples_split = 3,
                                n_estimators = 300)

gsearch2 = GridSearchCV(
    estimator= extree2,
    param_grid= params2,
    scoring='f1_weighted',
    n_jobs=-1,
    cv=gkf2,
    verbose=3,
)

extree_model2 = gsearch2.fit(X=X_train, y=y_train)

final_model = gsearch.best_estimator_
(gsearch2.best_params_, gsearch2.best_score_)


# In[87]:


extree_tuned = ExtraTreesClassifier(ccp_alpha = 0.0,
                                criterion = 'gini',
                                min_samples_split = 2,
                                class_weight = 'balanced',
                                max_depth = 15,
                                n_estimators = 400)

extree_tuned.fit(X_train, y_train)
y_pred_tuned = extree_tuned.predict(X_test)


# In[88]:


y_pred_tuned 


# Explalinable AI

# In[89]:


get_ipython().system('pip install shap')


# In[90]:


shap.initjs()


# In[91]:


X_sample = X_train.sample(100)
X_sample


# In[92]:


shap_values = shap.TreeExplainer(extree_tuned).shap_values(X_sample)


# In[93]:


shap.summary_plot(shap_values, X_sample, plot_type="bar")


# In[96]:


shap.summary_plot(shap_values, X_sample, max_display=29)


# In[97]:


shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0],
                shap_values[0][:], 
                X_sample)


# In[98]:


print(y_pred_tuned[50])
shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0], shap_values[1][50], X_sample.iloc[50])


# In[99]:


i=13
print(y_pred_tuned[i])
shap.force_plot(shap.TreeExplainer(extree_tuned).expected_value[0], shap_values[0][i], X_sample.values[i], feature_names = X_sample.columns)


# In[100]:


print(y_pred_tuned[10])
row = 10
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                              base_values=shap.TreeExplainer(extree_tuned).expected_value[0], data=X_sample.iloc[row],  
                                         feature_names=X_sample.columns.tolist()))


# In[101]:


shap.dependence_plot('Day_of_week', shap_values[2], X_sample)


# In[102]:


shap.dependence_plot('Age_band_of_driver', shap_values[2], X_sample)


# In[103]:


print(y_pred_tuned[10])
shap.decision_plot(shap.TreeExplainer(extree_tuned).expected_value[0], 
                   shap_values[2][:10], 
                   feature_names=X_sample.columns.tolist())


# In[109]:


get_ipython().system('pip install joblib')


# In[110]:


import joblib
joblib.dump(extree_model2, 'extree_model2.sav')


# In[ ]:




