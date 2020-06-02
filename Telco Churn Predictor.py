#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all relevant modules
import pyodbc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = None
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
from PIL import  Image


# In[2]:


#odbc connector to ingest user data
with pyodbc.connect("DSN=Treasure Data Presto ODBC DSN", autocommit=True) as conn:
    df = pd.read_sql("select * from ws_bukasa_r.wc_churn_sample", conn)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#Convert data from object to datetime
from datetime import datetime

df['date_key'] = pd.to_datetime(df['date_key'])
df['last_rge_date'] = pd.to_datetime(df['last_rge_date'])
df['first_usage_date'] = pd.to_datetime(df['first_usage_date'])
df['first_recharge_date'] = pd.to_datetime(df['first_recharge_date'])
df['last_recharge_date'] = pd.to_datetime(df['last_recharge_date'])
df['activation_date'] = pd.to_datetime(df['activation_date'])
df['churn_date'] = pd.to_datetime(df['churn_date'])
df['last_usage_date'] = pd.to_datetime(df['last_usage_date'])


# In[6]:


#change error dates as all customers were registered after 2000-01-01
df.loc[df['date_key'] < '2000-01-01', 'date_key' ] = np.NaN
df.loc[df['last_rge_date'] < '2000-01-01', 'last_rge_date' ] = np.NaN
df.loc[df['first_usage_date'] < '2000-01-01', 'first_usage_date' ] = np.NaN
df.loc[df['first_recharge_date'] < '2000-01-01', 'first_recharge_date' ] = np.NaN
df.loc[df['last_recharge_date'] < '2000-01-01', 'last_recharge_date' ] = np.NaN
df.loc[df['activation_date'] < '2000-01-01', 'activation_date' ] = np.NaN
df.loc[df['churn_date'] < '2000-01-01', 'churn_date' ] = np.NaN
df.loc[df['last_usage_date'] < '2000-01-01', 'last_usage_date' ] = np.NaN


# In[8]:


df.head()


# In[9]:


#column rev_mtd_current is a comma separated values colun for revenue of each day of month - use to split
df=pd.concat([df,df.rev_mtd_current.str.split(',',expand=True)],1)
df.head()


# In[10]:


#fill na with NaN value 
df.fillna(value=pd.np.nan, inplace=True)


# In[11]:


#Convert all created new columns to float dtypes
df[1] = df[1].astype(float)
df[2] = df[2].astype(float)
df[3] = df[3].astype(float)
df[4] = df[4].astype(float)
df[5] = df[5].astype(float)
df[6] = df[6].astype(float)
df[7] = df[7].astype(float)
df[8] = df[8].astype(float)
df[9] = df[9].astype(float)
df[10] = df[10].astype(float)
df[11] = df[11].astype(float)
df[12] = df[12].astype(float)
df[13] = df[13].astype(float)
df[14] = df[14].astype(float)
df[15] = df[15].astype(float)
df[16] = df[16].astype(float)
df[17] = df[17].astype(float)
df[18] = df[18].astype(float)
df[19] = df[19].astype(float)
df[20] = df[20].astype(float)
df[21] = df[21].astype(float)
df[22] = df[22].astype(float)
df[23] = df[23].astype(float)
df[24] = df[24].astype(float)
df[25] = df[25].astype(float)
df[26] = df[26].astype(float)
df[27] = df[27].astype(float)
df[28] = df[28].astype(float)
df[29] = df[29].astype(float)
df[30] = df[30].astype(float)
df[31] = df[31].astype(float)




# In[12]:


#Convert all NaN to 0 value
df[1] = df[1].fillna(0)
df[2] = df[2].fillna(0)
df[3] = df[3].fillna(0)
df[4] = df[4].fillna(0)
df[5] = df[5].fillna(0)
df[6] = df[6].fillna(0)
df[7] = df[7].fillna(0)
df[8] = df[8].fillna(0)
df[9] = df[9].fillna(0)
df[10] = df[10].fillna(0)
df[11] = df[11].fillna(0)
df[12] = df[12].fillna(0)
df[13] = df[13].fillna(0)
df[14] = df[14].fillna(0)
df[15] = df[15].fillna(0)
df[16] = df[16].fillna(0)
df[17] = df[17].fillna(0)
df[18] = df[18].fillna(0)
df[19] = df[19].fillna(0)
df[20] = df[20].fillna(0)
df[21] = df[21].fillna(0)
df[22] = df[22].fillna(0)
df[23] = df[23].fillna(0)
df[24] = df[24].fillna(0)
df[25] = df[25].fillna(0)
df[26] = df[26].fillna(0)
df[27] = df[27].fillna(0)
df[28] = df[28].fillna(0)
df[29] = df[29].fillna(0)
df[30] = df[30].fillna(0)
df[31] = df[31].fillna(0)


# In[13]:


#create new column to count how many days there was revenue data used in month
isUsed = lambda x:int(x > 0)
countdaysused = lambda row: isUsed(row[1]) + isUsed(row[2]) + isUsed(row[3]) + isUsed(row[4]) + isUsed(row[5]) + isUsed(row[6]) + isUsed(row[7]) + isUsed(row[8]) + isUsed(row[9]) + isUsed(row[10]) + isUsed(row[11]) + isUsed(row[12]) + isUsed(row[13]) + isUsed(row[14]) + isUsed(row[15]) + isUsed(row[16]) + isUsed(row[17]) + isUsed(row[18]) + isUsed(row[19]) + isUsed(row[20]) + isUsed(row[21]) + isUsed(row[22]) + isUsed(row[23]) + isUsed(row[24]) + isUsed(row[25]) + isUsed(row[26]) + isUsed(row[27]) + isUsed(row[28]) + isUsed(row[29]) + isUsed(row[30]) + isUsed(row[31])

df['days_used'] = df.apply(countdaysused,axis=1)


# In[14]:


df.head()


# In[15]:


# create new dataframe to aggregate relevant columns e.g. max min dates and time, means etc..
d = {'date_key':['min','max'],'last_rge_date':['min','max'],'first_usage_date':['min','max'],'last_usage_date':['min','max'],     'first_recharge_date':['min','max'],'last_recharge_date':['min','max'],'activation_date':['min','max'],     'churn_date':'max', 'aspu':'mean', 'voi_onnet_in_secs':'mean', 'voi_offnet_in_secs':'mean','voi_onnet_out_b_secs':'mean',    'voi_offnet_out_b_secs':'mean', 'voi_offnet_out_nb_secs':'mean','data_mb':'mean','rch_count_digital':'mean',     'rch_digital_rev': 'mean','rch_count_voucher':'mean', 'rch_voucher_rev':'mean', 'rch_airtime_amt':'mean',    'days_used': 'mean'}

res = df.groupby('msisdn').agg(d).reset_index()

res.columns = ['_'.join(col).strip() for col in res.columns.values]


# In[16]:


#create another dataframe to join on res to get latest status of each customer
df_late_sts = df[df['date_key'] == '2020-04-30']


# In[17]:


df_late_sts = df_late_sts.rename(columns={'msisdn':'msisdn_'})
df_late_sts.head()


# In[18]:


# join res and df_late_sts on 'msisdn_'
df_user = res.merge( df_late_sts[['msisdn_','status', 'price_plan_name', 'hs_make', 'smartphone_ind']], on='msisdn_', how='left')


# In[19]:


pd.set_option('display.max_columns', 500)
df_user.head(20)


# In[20]:


df_user.loc[df_user['status']  != 'CHURNED', 'churn_date_max' ] = np.NaN


# In[21]:


#feature engineer columns to assist modeling target
df_user['tenure'] = ((df_user.date_key_max - df_user.activation_date_min)/np.timedelta64(1,'M'))
df_user['usage_tenure'] = ((df_user.date_key_max - df_user.date_key_min)/np.timedelta64(1,'M'))
df_user['recharge_recency_days'] = ((df_user.churn_date_max.fillna(df_user.date_key_max) - df_user.last_rge_date_max)/np.timedelta64(1,'D'))


# In[22]:


df_user.head()


# In[23]:


df_user.info()


# In[24]:


#Uneven distribution of target variable - use SMOTE to sort balance further on
df_user['status'].value_counts()


# In[25]:


#Churn and Churned needs to be combned
df_user['status'] = df_user['status'].replace('CHURN', 'CHURNED')


# In[26]:


#Create separate df for each categorical target type for EDA
churn = df_user[df_user['status'] == 'CHURNED']
inactive = df_user[df_user['status'] == 'INACTIVE']
dormant = df_user[df_user['status'] == 'DORMANT']
active = df_user[df_user['status'] == 'ACTIVE']

target = df_user['status']
cat_cols = df_user[['price_plan_name', 'hs_make', 'smartphone_ind']]
num_cols = df_user[['usage_tenure', 'recharge_recency_days', 'tenure', 'days_used_mean', 'rch_airtime_amt_mean',              'rch_voucher_rev_mean', 'rch_count_voucher_mean', 'rch_digital_rev_mean', 'rch_count_digital_mean',              'data_mb_mean', 'voi_offnet_out_nb_secs_mean', 'voi_offnet_out_b_secs_mean', 'voi_onnet_out_b_secs_mean',              'voi_offnet_in_secs_mean', 'voi_onnet_in_secs_mean', 'aspu_mean']]


# In[27]:


#Create Pie chart to show percentage of target split
labels = df_user['status'].value_counts().keys().tolist()
values = df_user['status'].value_counts().values.tolist()

trace = go.Pie(labels=labels,
              values=values,
               marker=dict(colors = ['green', 'yelow', 'orange', 'red'],
                          line = dict(color = 'white',
                                     width = 1.3)
                          ),
               rotation = 90,
               hoverinfo = 'label+value+text',
               hole = .5)

layout = go.Layout(dict(title = 'Customer attrition in data',
                       plot_bgcolor = 'rgb(243,243,243)'
                       )
                  )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[28]:


#function  for pie plot for customer attrition types
def plot_pie(column) :
    
    trace1 = go.Pie(values = churn[column].value_counts().values.tolist(),
                   labels = churn[column].value_counts().keys().tolist(),
                   hoverinfo = 'label+percent+name',
                   domain = dict(x = [0,.25], y = [0,.75]),
                   name = 'Churned Customers',
                   marker = dict(line = dict(width = 2,
                                            color = 'rgb(243, 243, 243)')
                                ),
                   hole = .6
                   )
    trace2 = go.Pie(values = inactive[column].value_counts().values.tolist(),
                   labels = inactive[column].value_counts().keys().tolist(),
                   hoverinfo = 'label+percent+name',
                   domain = dict(x = [.25,.50], y = [0,.75]),
                   name = 'Inactive Customers',
                   marker = dict(line = dict(width = 2,
                                            color = 'rgb(243, 243, 243)')
                                ),
                   hole = .6
                   )
    trace3 = go.Pie(values = dormant[column].value_counts().values.tolist(),
                   labels = dormant[column].value_counts().keys().tolist(),
                   hoverinfo = 'label+percent+name',
                   domain = dict(x = [0.5,.75], y = [0,.75]),
                   name = 'Dormant Customers',
                   marker = dict(line = dict(width = 2,
                                            color = 'rgb(243, 243, 243)')
                                ),
                   hole = .6
                   )
    trace4 = go.Pie(values = active[column].value_counts().values.tolist(),
                   labels = active[column].value_counts().keys().tolist(),
                   hoverinfo = 'label+percent+name',
                   name = 'Active Customers',
                   domain = dict(x = [.75,1], y = [0,.75],),
                   marker = dict(line = dict(width = 2,
                                            color = 'rgb(243, 243, 243)')
                                ),
                   hole = .6
                   )
    
    layout = go.Layout(dict(title = column + ' distribution in customer attrition',
                      plot_bgcolor = 'rgb(243, 243, 243)',
                      paper_bgcolor = 'rgb(243, 243, 243)',
                      annotations = [dict(text = 'Churned Customers',
                                        font = dict(size = 9),
                                        showarrow = False,
                                        x = .06, y = .35),
                                    dict(text = 'Inactive Customers',
                                        font = dict(size = 9),
                                        showarrow = False,
                                        x = .37, y = .35),
                                    dict(text = 'Dormant Customers',
                                        font = dict(size = 9),
                                        showarrow = False,
                                        x = .67, y = .35),
                                    dict(text = 'Active Customers',
                                        font = dict(size = 9),
                                        showarrow = False,
                                        x = .925, y = .35)
                                    ]
                           ),
                      showlegend=False)
    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data= data, layout=layout)
    py.iplot(fig)


# In[29]:


#for all categorical columns plot pie
for i in cat_cols:
    plot_pie(i)


# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


#for numerical cols EDA apply
scaled_features = df_user.copy()
col_names = ['usage_tenure', 'recharge_recency_days', 'tenure', 'days_used_mean',             'rch_airtime_amt_mean', 'rch_voucher_rev_mean',             'rch_count_voucher_mean', 'rch_digital_rev_mean',             'rch_count_digital_mean', 'data_mb_mean', 'voi_offnet_out_nb_secs_mean',             'voi_offnet_out_b_secs_mean', 'voi_onnet_out_b_secs_mean',             'voi_offnet_in_secs_mean', 'voi_onnet_in_secs_mean', 'aspu_mean']

# features = scaled_features[col_names]
# scaler = StandardScaler().fit(features.values)
# features = scaler.transform(features.values)

# scaled_features[col_names] = features


# In[32]:


#function for histogram for customer attrition types
def histogram(column):
    trace1 = go.Histogram(x  = churn[column],
#                           histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .2 
                         )
    
    trace2 = go.Histogram(x  = inactive[column],
#                           histnorm= "percent",
                          name = "Inactive Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "white"
                                                    )
                                        ),
                         opacity = .5 
                         )
    
    trace3 = go.Histogram(x  = dormant[column],
#                           histnorm= "percent",
                          name = "Dormant Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "red"
                                                    )
                                        ),
                         opacity = .7 
                         )
    
    trace4 = go.Histogram(x  = active[column],
#                           histnorm= "percent",
                          name = "Active Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "blue"
                                                    )
                                        ),
                         opacity = .9 
                         )
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "count",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    
    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# In[33]:


for i in num_cols :
    histogram(i)


# In[34]:


df_user['msisdn_'] = df_user['msisdn_'].astype(float)


# In[35]:


#drop all datetime features for model development
clean_df = df_user.drop([ 'date_key_min', 'date_key_max', 'last_rge_date_min',                         'last_usage_date_max', 'first_recharge_date_min', 'first_recharge_date_max',                         'last_recharge_date_min', 'last_recharge_date_max', 'activation_date_min',                         'activation_date_max', 'churn_date_max', 'last_rge_date_max', 'first_usage_date_min',                        'first_usage_date_max', 'last_usage_date_min'], axis=1)


# In[36]:


clean_df.info()


# In[37]:


#Get dummies/ one hot encode categorical features
clean_df = pd.get_dummies(clean_df, prefix=['price_plan_name', 'smartphone_ind'], columns=['price_plan_name', 'smartphone_ind'])


# In[38]:


clean_df.info()


# In[39]:


cols_type = clean_df.dtypes != 'object'
inds = cols_type.index
numeric_cols = []
for i, col in enumerate(cols_type):
    if col:
        numeric_cols.append(inds[i])


# In[40]:


numeric_cols


# In[41]:


#for numerical cols minimize wide variations distributions
scaled_features = clean_df.copy()
col_names = numeric_cols
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[col_names] = features


# In[42]:


scaled_features.drop('hs_make', axis = 1, inplace=True)


# In[43]:


#manual encode to keep sinlge column
scaled_features.loc[scaled_features['status'] == 'ACTIVE', 'status' ] = 0
scaled_features.loc[scaled_features['status'] == 'CHURNED', 'status' ] = 1
scaled_features.loc[scaled_features['status'] == 'DORMANT', 'status' ] = 2
scaled_features.loc[scaled_features['status'] == 'INACTIVE', 'status' ] = 3

#change type
scaled_features['status'] = scaled_features['status'].astype(int)


# In[44]:


#import package to build models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score
from yellowbrick.classifier import DiscriminationThreshold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder


# In[45]:


#splitting train and test
train, test = train_test_split(scaled_features, test_size=0.25, random_state=111)

#customer id col
Id_col     = ['msisdn_']
#Target columns
target_col = ["status"]
#hs
hs_model = ['hs_make']


# In[46]:


#separating dependent and indipendent variable
cols = [i for i in scaled_features.columns if i not in Id_col + target_col + hs_model]
train_X = train[cols]
train_Y = train[target_col]
test_X = test[cols]
test_Y = test[target_col]


# In[47]:


#replace NaN with 0
train_X = train_X.fillna(0)
test_X = test_X.fillna(0)


# In[48]:


train_X = train_X.dropna()
test_X = test_X.dropna()


# In[49]:


def telecom_churn_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf,threshold_plot) :
    
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
        
    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))

    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)

    #plot confusion matrix
    trace1 = go.Heatmap(z = conf_matrix ,
                        x = ["Active","Chruned", "Dormant", "Inactive"],
                        y = ["Active","Chruned", "Dormant", "Inactive"],
                        showscale  = False,colorscale = "Picnic",
                        name = "matrix")
    
    
    #plot coeffs
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Picnic",
                                  line = dict(width = .6,color = "black")))
    
    #subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
#                                             'Receiver operating characteristic',
                                            'Feature Importances'))
    
    fig.append_trace(trace1,1,1)
#     fig.append_trace(trace2,1,2)
#     fig.append_trace(trace3,1,2)
    fig.append_trace(trace4,2,1)
    
    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig)


# In[50]:


logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='multinomial', n_jobs=1,
          penalty='l2', random_state=None, tol=0.0001,
          verbose=0, warm_start=False)

telecom_churn_prediction(logit,train_X,test_X,train_Y,test_Y,cols,"coefficients",threshold_plot = True)


# In[51]:


d = {'recharge_recency_days': 'mean'}

clean_df.groupby('status').agg(d)


# In[52]:


plt.figure(figsize=(12,6))
clean_df[clean_df['status'] == 'ACTIVE']['recharge_recency_days'].hist(alpha =.5, color='green',
                                                                            bins=5, label='Actvie')
clean_df[clean_df['status'] == 'CHURNED']['recharge_recency_days'].hist(alpha =.5, color='red',
                                                                            bins=5, label='Churned')
clean_df[clean_df['status'] == 'DORMANT']['recharge_recency_days'].hist(alpha = .5, color='orange',
                                                                             bins=5, label='Dormant')
# clean_df[clean_df['status'] == 'INACTIVE']['recharge_recency_days'].hist(alpha = .5, color='blue',
#                                                                              bins=5, label='Inactive')

plt.legend()
plt.xlabel('Recharge_recency_days')


# In[ ]:




