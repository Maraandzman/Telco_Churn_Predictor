# Telco_Churn_Predictor
Telco Churn Prediction Model
---
**Context:**

A Telco business has wants to establish a prediction model to help assist their markeing team to initiate a campaign to mitigate quantity of customers churning or being inactive. 

The data provided of 100 000 distinct customers, and their respective usage behaviour over the course of 2+/- years, and 

**Project Objective:**

The target feature is a multi-class. "ACTIVE", "CHURNED", "DORMANT", "INACTIVE".
predict is status with relative good accuracy. 

# Code and Resources
---                               
**Python Version**: 3.7

**Package**: pandas, numpy, matplotlib, plotly, logisticregression

# Data

0   customer_id                  object 
 1   status                  object 
 2   price_plan_name         object 
 3   hs_make                 object 
 4   smartphone_ind          int64  
 5   date_key                object 
 6   months                  object 
 7   last_rge_date           object 
 8   first_usage_date        object 
 9   last_usage_date         object 
 10  first_recharge_date     object 
 11  last_recharge_date      object 
 12  activation_date         object 
 13  churn_date              object 
 14  aspu                    float64
 15  voi_onnet_in_secs       int64  
 16  voi_offnet_in_secs      int64  
 17  voi_onnet_out_b_secs    int64  
 18  voi_onnet_out_nb_secs   int64  
 19  voi_offnet_out_b_secs   int64  
 20  voi_offnet_out_nb_secs  int64  
 21  data_mb                 float64
 22  rch_count_digital       int64  
 23  rch_digital_rev         float64
 24  rch_count_voucher       int64  
 25  rch_voucher_rev         float64
 26  rch_airtime_amt         float64
 27  usage_count             float64
 28  recharge_count          float64
 29  rev_mtd_current         object 
 
 # EDA
