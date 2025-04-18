#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HW1.py
2025-04-10
Python implementation of the R analysis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.outliers_influence import summary_table
import wooldridge

# (c1)
print("#(c1)")
print("#(i)Compute first order autocorrelation")

# Load hseinv dataset
hseinv = wooldridge.data('hseinv')

# Calculate first order autocorrelation
print("ACF for linvpc:")
linvpc_acf = acf(hseinv['linvpc'], nlags=1)
print(f"0: {linvpc_acf[0]:.3f}, 1: {linvpc_acf[1]:.3f}")

print("ACF for lprice:")
lprice_acf = acf(hseinv['lprice'], nlags=1)
print(f"0: {lprice_acf[0]:.3f}, 1: {lprice_acf[1]:.3f}")

# Linearly detrend log(invpc)
hseinv['time'] = np.arange(1, len(hseinv) + 1)
lm_invpc = sm.OLS(hseinv['linvpc'], sm.add_constant(hseinv['time'])).fit()
hseinv['detrended_log_invpc'] = lm_invpc.resid

# Linearly detrend log(price)
lm_price = sm.OLS(hseinv['lprice'], sm.add_constant(hseinv['time'])).fit()
hseinv['detrended_log_price'] = lm_price.resid

# Compute first order autocorrelation after detrending
print("ACF for detrended_log_invpc:")
detrended_log_invpc_acf = acf(hseinv['detrended_log_invpc'], nlags=1)
print(f"0: {detrended_log_invpc_acf[0]:.3f}, 1: {detrended_log_invpc_acf[1]:.3f}")

print("ACF for detrended_log_price:")
detrended_log_price_acf = acf(hseinv['detrended_log_price'], nlags=1)
print(f"0: {detrended_log_price_acf[0]:.3f}, 1: {detrended_log_price_acf[1]:.3f}")

# (ii) Estimate the equation
print("\n#(ii) Estimate the equation")
hseinv['diff_log_price'] = hseinv['lprice'].diff()
model1 = sm.OLS(hseinv['linvpc'].iloc[1:], 
               sm.add_constant(hseinv[['diff_log_price', 'time']].iloc[1:])).fit()
print(model1.summary())

# (iii) Use detrended log(invpc) as dependent variable
print("\n#(iii) Use detrended log(invpc) as dependent variable")
model2 = sm.OLS(hseinv['detrended_log_invpc'].iloc[1:], 
               sm.add_constant(hseinv[['diff_log_price', 'time']].iloc[1:])).fit()
print(model2.summary())

# (iv) Use first difference of log(invpc) as dependent variable
print("\n#(iv) Use first difference of log(invpc) as dependent variable")
hseinv['diff_log_invpc'] = hseinv['linvpc'].diff()
model3 = sm.OLS(hseinv['diff_log_invpc'].iloc[1:], 
               sm.add_constant(hseinv['diff_log_price'].iloc[1:])).fit()
print(model3.summary())

# (C4)
print("\n#(C4)")
print("#(i)Estimate a Phillips curve where inflation and unemployment are in first differences")

# Load phillips dataset
phillips = wooldridge.data('phillips')

# Create time series data starting from 1948
phillips.index = pd.date_range(start='1948', periods=len(phillips), freq='A')

# Calculate first differences
phillips['d_inf'] = phillips['inf'].diff()
phillips['d_unem'] = phillips['unem'].diff()

# Filter data up to 2003
phillips_2003 = phillips[phillips.index.year <= 2003]

# Estimate the model
reg_ea = sm.OLS(phillips_2003['d_inf'].dropna(), 
               sm.add_constant(phillips_2003['d_unem'].dropna())).fit()
print(reg_ea.summary())

# (ii) Comparison of both Phillips curve models
print("\n#(ii)Comparison of both Phillips curve models")
reg_ea2 = sm.OLS(phillips_2003['d_inf'].dropna(), 
                sm.add_constant(phillips_2003['unem'].iloc[1:])).fit()
print(reg_ea2.summary())

# (C8)
print("\n#(C8)")
print("#(i)Obtain AR(1) for unemployment rate and make a prediction for that of 2004.")

# Create lag variables
phillips['unem_1'] = phillips['unem'].shift(1)
phillips['inf_1'] = phillips['inf'].shift(1)

# Filter data up to 2003
phillips_2003 = phillips[phillips.index.year <= 2003]

# AR(1) model for unemployment
model1 = sm.OLS(phillips_2003['unem'].iloc[1:], 
               sm.add_constant(phillips_2003['unem_1'].iloc[1:])).fit()

# Make prediction for 2004
# Get the last unemployment rate (2003) to predict 2004
unem_2003 = phillips[phillips.index.year == 2003]['unem'].values[0]
pred_data = pd.DataFrame({'const': [1], 'unem_1': [unem_2003]})
f1 = model1.predict(pred_data)
print(f"Prediction for 2004 using AR(1): {f1.values[0]:.6f}")

# Actual 2004 unemployment rate (from external source)
actual_2004 = 5.5  # The actual 2004 unemployment rate
print(f"Actual 2004 unemployment rate: {actual_2004:.1f}%")
print(f"Prediction error: {f1.values[0] - actual_2004:.6f}")

# (ii) Add lag inflation rate into the model
print("\n#(ii)Add lag inflation rate into the model")
model2 = sm.OLS(phillips_2003['unem'].iloc[1:], 
               sm.add_constant(phillips_2003[['unem_1', 'inf_1']].iloc[1:])).fit()

# Print summary of both models
print("\nModel 1 (AR(1)):")
print(f"Observations: {model1.nobs}")
print(f"Adjusted R-squared: {model1.rsquared_adj:.3f}")
print(f"Residual Std. Error: {np.sqrt(model1.scale):.3f}")
print("\nModel 2 (AR(1) with inflation):")
print(f"Observations: {model2.nobs}")
print(f"Adjusted R-squared: {model2.rsquared_adj:.3f}")
print(f"Residual Std. Error: {np.sqrt(model2.scale):.3f}")

# (iii) Again make a prediction using modified model
print("\n#(iii)Again make a prediction using modified model")
# Get the last unemployment and inflation rates (2003) to predict 2004
unem_2003 = phillips[phillips.index.year == 2003]['unem'].values[0]
inf_2003 = phillips[phillips.index.year == 2003]['inf'].values[0]
pred_data2 = pd.DataFrame({'const': [1], 'unem_1': [unem_2003], 'inf_1': [inf_2003]})
f2 = model2.predict(pred_data2)
print(f"Prediction for 2004 using modified model: {f2.values[0]:.6f}")
print(f"Actual 2004 unemployment rate: {actual_2004:.1f}%")
print(f"Prediction error: {f2.values[0] - actual_2004:.6f}")

# (iv) Construct a five-year 95% prediction interval for new observations and check the range
print("\n#(iv)Construct a five-year 95% prediction interval for new observations and check the range")
# Make predictions with intervals for 1999-2003
for year in range(1999, 2004):
    # Get the previous year's unemployment rate
    prev_year = year - 1
    unem_prev = phillips[phillips.index.year == prev_year]['unem'].values[0]
    
    # Create prediction data
    pred_data = pd.DataFrame({'const': [1], 'unem_1': [unem_prev]})
    
    # Get prediction with intervals
    prediction = model1.get_prediction(pred_data)
    frame = prediction.summary_frame(alpha=0.05)
    
    print(f"{year}: fit={frame['mean'].values[0]:.6f}, lwr={frame['mean_ci_lower'].values[0]:.6f}, upr={frame['mean_ci_upper'].values[0]:.6f}")
