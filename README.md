# Telco_Churn_Predictor
Telco Churn Prediction Model
---
**Context:**

A Telco business has wants to establish a prediction model to help assist their markeing team to initiate a campaign to mitigate quantity of customers churning or being inactive. 

The data provided of 100 000 distinct customers, and their respective usage behaviour over the course of 2+/- years.

**Project Objective:**

The target feature is a multi-class labels "ACTIVE", "CHURNED", "DORMANT", "INACTIVE".
predict status with relative good accuracy. 

# Code and Resources
---                               
**Python Version**: 3.7

**Package**: pandas, numpy, matplotlib, plotly, logisticregression

# Data

 -  customer_id                  object 
 -   status                  object 
 -   price_plan_name         object 
 -   hs_make                 object 
 -   smartphone_ind          int64  
 -   date_key                object 
 -   months                  object 
 -   last_rge_date           object 
 -   first_usage_date        object 
 -   last_usage_date         object 
 -  first_recharge_date     object 
 -  last_recharge_date      object 
 -  activation_date         object 
 -  churn_date              object 
 -  aspu                    float64
 -  voi_onnet_in_secs       int64  
 -  voi_offnet_in_secs      int64  
 -  voi_onnet_out_b_secs    int64  
 -  voi_onnet_out_nb_secs   int64  
 -  voi_offnet_out_b_secs   int64  
 -  voi_offnet_out_nb_secs  int64  
 -  data_mb                 float64
 -  rch_count_digital       int64  
 -  rch_digital_rev         float64
 -  rch_count_voucher       int64  
 -  rch_voucher_rev         float64
 -  rch_airtime_amt         float64
 -  usage_count             float64
 -  recharge_count          float64
 -  rev_mtd_current         object 
 
 # EDA
![](Customer_Attrition_Data.png)

![](Price_Plan_distro.png)

![](Smartphone_Indicator.png)

![](Tenure_Distro_in_Customer_Attrition.png)

![](Days_used_mean.png)

# Model

Kept a simple will performaing logistic regession classifier model:

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='multinomial', n_jobs=1, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
                   
![](Classification_Report.PNG)

![](Model_performance.png)

![](Recharge_recency_Attrition_Group.PNG)

![](Recharge_recency_days_Hist.png)


