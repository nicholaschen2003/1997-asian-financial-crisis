import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import grangercausalitytests

# --------------------STEP 1: Collecting data---------------------------

# Nice visual of gdp per capita comparison:
# https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=HK-SG
# Unit: Current USD
did_per_capita_gdp = pd.read_csv('raw_data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_631205.csv')

# Contracts Awarded And Progress Payments Certified By Sector And Development Type (Singapore):
# https://tablebuilder.singstat.gov.sg/table/TS/M400221
# Unit: Millions of SGD
sg_construction = pd.read_csv('raw_data/M400221.csv')

# Gross Domestic Product At Current Prices, By Industry (SSIC 2020) (Singapore):
# https://tablebuilder.singstat.gov.sg/table/TS/M015731
# Unit: Millions of SGD
sg_gdp = pd.read_csv('raw_data/M015731.csv')

# Research And Development Expenditure By Area Of Research (Singapore):
# https://tablebuilder.singstat.gov.sg/table/TS/M081331
# Unit: Millions of SGD
sg_rd = pd.read_csv('raw_data/M081331.csv')

# Table 615-66001 : Gross value of construction works performed by main contractors (Hong Kong):
# https://www.censtatd.gov.hk/en/web_table.html?id=615-66001#
# Unit: Millions of HKD
hk_construction = pd.read_csv('raw_data/Table 615-66001_en.csv')

# Table 710-86001 : Gross domestic expenditure on R&D by performing sector (Hong Kong):
# https://www.censtatd.gov.hk/en/web_table.html?id=710-86001#
# Unit: %
hk_rd = pd.read_csv('raw_data/Table 710-86001_en.csv')

# Table 310-31001 : Gross Domestic Product (GDP), implicit price deflator of GDP and per capita GDP (Hong Kong):
# https://www.censtatd.gov.hk/en/web_table.html?id=31#
# Unit: %
hk_per_capita_gdp_growth = pd.read_csv('raw_data/Table 310-31001_en.csv')

# --------------------STEP 2: Clean and visualize data---------------------------

# this part will just graph gdp per captia (and differences) between sg and hk and argue
# that the years 1997-2003 are sort of a 'treatment period' (think asian financial crisis, handover of hk, sars outbreak, etc.)
# and that the policy decisions in response to these events in each country resulted in the differenct trajectories

# diff-in-diff visual for gdp per capita vs time
# print(did_per_capita_gdp.iloc[96])
# print(did_per_capita_gdp.iloc[208])
did = pd.DataFrame({'Hong Kong': did_per_capita_gdp.iloc[96][4:-5].astype('float64'),
                    'Singapore': did_per_capita_gdp.iloc[208][4:-5].astype('float64')})
did.index = list(range(1960, 2019))
did.index.name = 'Year'
did.to_csv('cleaned_data/gdp_per_capita.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=did, marker='.', dashes=False, markeredgecolor='black')
plt.title('Singapore vs. Hong Kong GDP Per Capita Over Time')
plt.xlabel('Year')
plt.ylabel('GDP Per Capita (Current USD)')
plt.xticks(np.arange(1960, 2019, 5))
plt.axvline(x=1997, ymin=0, ymax=1, linestyle='dashed', color='green')
plt.axvline(x=2003, ymin=0, ymax=1, linestyle='dashed', color='green')
plt.text(1995.8,50000,'1997',rotation=90, color='green')
plt.text(2003.1,50000,'2003',rotation=90, color='green')
plt.axvspan(1997, 2003, alpha=0.25, color='green')
plt.savefig('output/diff-in-diff.png', dpi=300)
plt.show()

# extra visual for visuaizing diff between two countries
did_diff = pd.DataFrame({'Difference': did_per_capita_gdp.iloc[96][4:-5] - did_per_capita_gdp.iloc[208][4:-5]})
did_diff.index = list(range(1960, 2019))
did_diff.index.name = 'Year'
did_diff.to_csv('cleaned_data/diff_in_gdp_per_capita.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=did_diff, marker='.', markeredgecolor='black')
plt.title('Difference in Singapore Hong Kong GDP Per Capita Over Time')
plt.xlabel('Year')
plt.ylabel('Difference in GDP Per Capita (Current USD)')
plt.xticks(np.arange(1960, 2019, 5))
plt.axvline(x=1997, ymin=0, ymax=1, linestyle='dashed', color='green')
plt.axvline(x=2003, ymin=0, ymax=1, linestyle='dashed', color='green')
plt.text(1995.8,-12500,'1997',rotation=90, color='green')
plt.text(2003.1,-12500,'2003',rotation=90, color='green')
plt.axvspan(1997, 2003, alpha=0.25, color='green')
plt.savefig('output/diffrences.png', dpi=300)
plt.show()

# truncate datasets to start at year 2000 (earliest common year) and end at year 2018 (avoid covid)
# this part will be looking at effects of investment in r&d vs. investment in real estate
# normalize everything to share of gdp (at current prices)

# print(sg_construction.iloc[28][6:25])
# print(sg_gdp.iloc[0][6:25])
# print(sg_construction.dtypes)
# print(sg_gdp.dtypes)
sg_construction_share = pd.DataFrame({'Singapore Construction Share of GDP (%)': 
                                      sg_construction.iloc[28][6:25] / sg_gdp.iloc[0][6:25].astype('float64') * 100}).iloc[::-1]
# print(sg_construction_share)

# print(sg_rd.iloc[0][4:23])
sg_rd_share = pd.DataFrame({'Singapore R&D Share of GDP (%)': 
                            sg_rd.iloc[0][4:23] / sg_gdp.iloc[0][6:25].astype('float64') * 100}).iloc[::-1]
# print(sg_rd_share)

sg_rd_share_1999 = pd.DataFrame({'Singapore R&D Share of GDP (%)': 
                                 sg_rd.iloc[0][4:24] / sg_gdp.iloc[0][6:26].astype('float64') * 100}).iloc[::-1]

# print(hk_per_capita_gdp_growth.iloc[0][41:60])
hk_construction = hk_construction[['In nominal terms']][2:21].T
hk_construction.columns = [str(x) for x in range(2000, 2019)]
# print(hk_construction.iloc[0])
hk_construction_share = pd.DataFrame({'Hong Kong Construction Share of GDP (%)': 
                                      hk_construction.iloc[0].astype('float64') / hk_per_capita_gdp_growth.iloc[0][41:60].astype('float64') * 100})
# print(hk_construction_share)

hk_rd_share = hk_rd[['Ratio of gross domestic expenditure on R&D to GDP (2)']][5:24]
hk_rd_share.index = [str(x) for x in range(2000, 2019)]
hk_rd_share.columns = ['Hong Kong R&D Share of GDP (%)']
# print(hk_rd_share)

hk_rd_share_1999 = hk_rd[['Ratio of gross domestic expenditure on R&D to GDP (2)']][4:24]
hk_rd_share_1999.index = [str(x) for x in range(1999, 2019)]
hk_rd_share_1999.columns = ['Hong Kong R&D Share of GDP (%)']
# print(hk_rd_share)

# construction share of gdp
construction_share = pd.concat([hk_construction_share.astype('float64'), sg_construction_share.astype('float64')], axis=1)
construction_share.index = list(range(2000,2019))
construction_share.index.name = 'Year'
construction_share.to_csv('cleaned_data/construction_share.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=construction_share, marker='.', dashes=False, markeredgecolor='black')
plt.title('Singapore vs. Hong Kong Construction Share of GDP Over Time')
plt.xlabel('Year')
plt.ylabel('Share of GDP (%)')
plt.xticks(np.arange(2000, 2019, 2))
# plt.axvline(x=2003, ymin=0, ymax=1, linestyle='dashed', color='green')
# plt.text(2003.1,50000,'2003',rotation=90, color='green')
# plt.axvspan(2000, 2003, alpha=0.25, color='green')
plt.savefig('output/construction_share.png', dpi=300)
plt.show()

# r&d share of gdp
rd_share = pd.concat([hk_rd_share.astype('float64'), sg_rd_share.astype('float64')], axis=1)
rd_share.index = list(range(2000,2019))
rd_share.index.name = 'Year'
rd_share.to_csv('cleaned_data/rd_share.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=rd_share, marker='.', dashes=False, markeredgecolor='black')
plt.title('Singapore vs. Hong Kong R&D Share of GDP Over Time')
plt.xlabel('Year')
plt.ylabel('Share of GDP (%)')
plt.xticks(np.arange(2000, 2019, 2))
# plt.axvline(x=2003, ymin=0, ymax=1, linestyle='dashed', color='green')
# plt.text(2003.1,50000,'2003',rotation=90, color='green')
# plt.axvspan(2000, 2003, alpha=0.25, color='green')
plt.savefig('output/rd_share.png', dpi=300)
plt.show()

# r&d share of gdp starting from 1999 (just in case, but it's not very different)
rd_share_1999 = pd.concat([hk_rd_share_1999.astype('float64'), sg_rd_share_1999.astype('float64')], axis=1)
rd_share_1999.index = list(range(1999,2019))
rd_share_1999.to_csv('cleaned_data/rd_share_1999.csv')
plt.figure(figsize=(10, 6))
sns.lineplot(data=rd_share_1999, marker='.', dashes=False, markeredgecolor='black')
plt.title('Singapore vs. Hong Kong R&D Share of GDP Over Time')
plt.xlabel('Year')
plt.ylabel('Share of GDP (%)')
plt.xticks(np.arange(1999, 2019, 2))
# plt.axvline(x=2003, ymin=0, ymax=1, linestyle='dashed', color='green')
# plt.text(2003.1,50000,'2003',rotation=90, color='green')
# plt.axvspan(2000, 2003, alpha=0.25, color='green')
plt.savefig('output/rd_share_1999.png', dpi=300)
plt.show()

# --------------------STEP 3: Stationary and cointegration tests, prep for VECM---------------------------

def adf_test(series,title=''):
    '''
    Pass in a time series and an optional title, returns an ADF report
    '''
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          # .to_string() removes the line 'dtype: float64'
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis')
        print('Reject the null hypothesis')
        print('Data has no unit root and is stationary')
    else:
        print('Weak evidence against the null hypothesis')
        print('Fail to reject the null hypothesis')
        print('Data has a unit root and is non-stationary')
    print('----------------------------------')

def johansen_test(df, det_order=0):
    result = coint_johansen(df, det_order, 1)
    traces = result.lr1
    cvts = result.cvt[:, 1]
    print('Johansen Test Statistics')
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(f'{col}: Statistic {trace} > Critical Value {cvt} -> {trace > cvt}')
    print('----------------------------------')

# merge into one df
data = did.join(construction_share, how='inner')
data = data.join(rd_share, how='inner')

# print(data)

# Performing ADF tests
adf_test(data['Hong Kong'], 'Hong Kong GDP per Capita') # non-stationary
adf_test(data['Singapore'], 'Singapore GDP per Capita') # non-stationary
adf_test(data['Hong Kong Construction Share of GDP (%)'], 'Hong Kong Construction Share') # stationary
adf_test(data['Singapore Construction Share of GDP (%)'], 'Singapore Construction Share') # stationary
adf_test(data['Hong Kong R&D Share of GDP (%)'], 'Hong Kong R&D Share') # stationary
adf_test(data['Singapore R&D Share of GDP (%)'], 'Singapore R&D Share') # non-stationary

# Differentiating the non-stationary series
data['Diff Hong Kong GDP per Capita'] = data['Hong Kong'].diff().dropna()
data['Diff Singapore GDP per Capita'] = data['Singapore'].diff().dropna()
data['Diff Singapore R&D Share of GDP (%)'] = data['Singapore R&D Share of GDP (%)'].diff().dropna()

# Check if differenced data is stationary
adf_test(data['Diff Hong Kong GDP per Capita'], 'Diff Hong Kong GDP per Capita') # non-stationary
adf_test(data['Diff Singapore GDP per Capita'], 'Diff Singapore GDP per Capita') # non-stationary
adf_test(data['Diff Singapore R&D Share of GDP (%)'], 'Diff Singapore R&D Share') # stationary

# Try HP-Filtering data
data['Cycle Hong Kong GDP per Capita'], data['Trend Hong Kong GDP per Capita'] = hpfilter(data['Hong Kong'], lamb=100)
data['Cycle Singapore GDP per Capita'], data['Trend Singapore GDP per Capita'] = hpfilter(data['Hong Kong'], lamb=100)

# Check if data is stationary
adf_test(data['Trend Hong Kong GDP per Capita'], 'Trend Hong Kong GDP per Capita') # non-stationary
adf_test(data['Trend Singapore GDP per Capita'], 'Trend Singapore GDP per Capita') # non-stationary

# Use percent change for gdp per capita values instead
data['Pct Change Hong Kong GDP per Capita'] = data['Trend Hong Kong GDP per Capita'].pct_change() * 100
data['Pct Change Singapore GDP per Capita'] = data['Trend Singapore GDP per Capita'].pct_change() * 100

# Check if data is stationary
adf_test(data['Pct Change Hong Kong GDP per Capita'], 'Pct Change Hong Kong GDP per Capita') # stationary
adf_test(data['Pct Change Singapore GDP per Capita'], 'Pct Change Singapore GDP per Capita') # stationary

# Performing cointegration tests
non_stationary_series = data[['Hong Kong', 'Singapore', 'Singapore R&D Share of GDP (%)']]
johansen_test(non_stationary_series) # Hong Kong GDP per Capita is cointegrated

# Prepare the dataset for VECM
vecm_data_hk = data.dropna()[['Pct Change Hong Kong GDP per Capita', 
                              'Hong Kong Construction Share of GDP (%)',
                              'Hong Kong R&D Share of GDP (%)']]

vecm_data_sg = data.dropna()[['Pct Change Singapore GDP per Capita', 
                              'Singapore Construction Share of GDP (%)', 
                              'Singapore R&D Share of GDP (%)']]

vecm_data_hk.columns = ['% Change HK GDP Per Capita', 'HK Construct. Share', 'HK R&D Share']
vecm_data_sg.columns = ['% Change SG GDP Per Capita', 'SG Construct. Share', 'SG R&D Share']

# get optimal lag lengths (VAR lag length - 1)
lag_hk = VAR(vecm_data_hk).select_order().aic - 1
lag_sg = VAR(vecm_data_sg).select_order().aic - 1

# Fit the VECM model
vecm_hk = VECM(vecm_data_hk, k_ar_diff=lag_hk, coint_rank=1)
vecm_fit_hk = vecm_hk.fit()
vecm_sg = VECM(vecm_data_sg, k_ar_diff=lag_sg, coint_rank=1)
vecm_fit_sg = vecm_sg.fit()

# Display the summary of the VECM model
print(vecm_fit_hk.summary())
print('##############################################################################')
print(vecm_fit_sg.summary())

# --------------------STEP 4: Impulse Response Functions---------------------------

irf_hk = vecm_fit_hk.irf(10)
irf_hk.plot(orth=False, figsize=(16,10))
plt.savefig('output/hk_irf.png', dpi=300)
plt.show()

irf_sg = vecm_fit_sg.irf(10)
irf_sg.plot(orth=False, figsize=(16,10))
plt.savefig('output/sg_irf.png', dpi=300)
plt.show()

# --------------------STEP 5: Granger Causality---------------------------

# Performing Granger causality tests for Hong Kong
maxlag = 2  # Define the maximum number of lags to test
test_results_hk = {}
for col in vecm_data_hk.columns[1:]:
    test_results_hk[col] = grangercausalitytests(vecm_data_hk[['% Change HK GDP Per Capita', col]], maxlag=maxlag, verbose=False)
    print(f'Granger Causality Test results for "% Change HK GDP Per Capita" caused by "{col}" in Hong Kong:')
    for key in test_results_hk[col]:
        test = test_results_hk[col][key][0]['ssr_ftest']
        print(f'  Lag {key}: F-statistic = {test[0]:.4f}, p-value = {test[1]:.4f}')

print('----------------------------------')

# Performing Granger causality tests for Singapore
maxlag = 2  # Define the maximum number of lags to test
test_results_sg = {}
for col in vecm_data_sg.columns[1:]:
    test_results_sg[col] = grangercausalitytests(vecm_data_sg[['% Change SG GDP Per Capita', col]], maxlag=maxlag, verbose=False)
    print(f'Granger Causality Test results for "% Change SG GDP Per Capita" caused by "{col}" in Singapore:')
    for key in test_results_sg[col]:
        test = test_results_sg[col][key][0]['ssr_ftest']
        print(f'  Lag {key}: F-statistic = {test[0]:.4f}, p-value = {test[1]:.4f}')