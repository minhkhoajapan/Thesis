import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

NUMBER_OF_BETA_PORTFOLIO = 30
STOCK_MISSING_VALUE = -99.9999
MARKET_MISSING_VALUE = -99.99

SPM_Data = pd.read_excel("SPM.xlsx", engine="openpyxl", header=None)
SPM_Data.columns = ['code', 'year', 'JunPrice', 'LnMV', 'BPR', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']

GPIN_txt = np.genfromtxt(
    'GPIN_results.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'gpin_a', 'gpin_p', 'gpin_eta', 'gpin_r', 'gpin_d', 'gpin_th', 'gpin_f', 'gpin_rc', 'GPIN']
)
GPIN_Data = pd.DataFrame(GPIN_txt)

Mrk_Data = pd.read_excel("risk_free.xlsx", engine="openpyxl", header=None)
Mrk_Data.columns = [
    'year',
    'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
    'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7', 'rf8', 'rf9', 'rf10', 'rf11', 'rf12'
]

preranking_beta = pd.DataFrame(np.genfromtxt(
    'preranking_beta.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'beta']
))

preranking_beta['beta_group'] = preranking_beta.groupby('year')['beta'].transform(
    lambda x: pd.qcut(x, q = NUMBER_OF_BETA_PORTFOLIO, labels = list(range(1, 31, 1)))
)
all_codes = preranking_beta['code'].unique().tolist()

portfolio_beta = pd.DataFrame(np.genfromtxt(
    'portfolio_beta.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['index', 'beta']
))

def cross_section_regression(year):
    betas = []
    for month in range(7, 13):
        X, y = [], []
        for code in all_codes:
            spm_row = SPM_Data[(SPM_Data['year'] == year) & (SPM_Data['code'] == code)]
            gpin_row = GPIN_Data[(GPIN_Data['year'] == year) & (GPIN_Data['code'] == code)]
            market_row = Mrk_Data[Mrk_Data['year'] == year]
            preranking_beta_row = preranking_beta[(preranking_beta['year'] == year) & (preranking_beta['code'] == code)]
            if spm_row.empty or gpin_row.empty or market_row.empty or preranking_beta_row.empty:
                continue
            
            monthly_return = spm_row.iloc[0].get(f'r{month}', np.nan)
            gpin = gpin_row.iloc[0].get('GPIN', np.nan)
            rf = market_row.iloc[0].get(f'rf{month}', np.nan)
            LnMV = spm_row.iloc[0].get('LnMV', np.nan)
            BPR = spm_row.iloc[0].get('BPR', np.nan)
            portfolio_num = preranking_beta_row.iloc[0].get('beta_group')
            if any(pd.isna(v) for v in [monthly_return, gpin, rf, LnMV, BPR, portfolio_num]) or monthly_return == STOCK_MISSING_VALUE or rf == MARKET_MISSING_VALUE or BPR <= 0:
                continue
            
            ingroup_beta = portfolio_beta[portfolio_beta['index'] == portfolio_num].iloc[0].get('beta')
            X.append([ingroup_beta, gpin, LnMV, round(math.log(BPR), 5)])
            y.append(monthly_return - rf)

        monthly_betas = LinearRegression().fit(X, y).coef_.tolist() 
        print(f'complete regression for year {year} month {month}')
        betas.append(monthly_betas)

    return betas

def cross_section_regression_next_year(year):
    betas = []
    for month in range(1, 7):
        X, y = [], []
        for code in all_codes:
            spm_row = SPM_Data[(SPM_Data['year'] == year) & (SPM_Data['code'] == code)]
            spm_row_next_year = SPM_Data[(SPM_Data['year'] == year + 1) & (SPM_Data['code'] == code)]
            gpin_row = GPIN_Data[(GPIN_Data['year'] == year) & (GPIN_Data['code'] == code)]
            market_row = Mrk_Data[Mrk_Data['year'] == year + 1]
            preranking_beta_row = preranking_beta[(preranking_beta['year'] == year) & (preranking_beta['code'] == code)]
            if spm_row.empty or spm_row_next_year.empty or gpin_row.empty or market_row.empty or preranking_beta_row.empty:
                continue
            
            monthly_return = spm_row_next_year.iloc[0].get(f'r{month}', np.nan)
            gpin = gpin_row.iloc[0].get('GPIN', np.nan)
            rf = market_row.iloc[0].get(f'rf{month}', np.nan)
            LnMV = spm_row.iloc[0].get('LnMV', np.nan)
            BPR = spm_row.iloc[0].get('BPR', np.nan)
            portfolio_num = preranking_beta_row.iloc[0].get('beta_group')
            if any(pd.isna(v) for v in [monthly_return, gpin, rf, LnMV, BPR, portfolio_num]) or monthly_return == STOCK_MISSING_VALUE or rf == MARKET_MISSING_VALUE or BPR <= 0:
                continue
            
            ingroup_beta = portfolio_beta[portfolio_beta['index'] == portfolio_num].iloc[0].get('beta')
            X.append([ingroup_beta, gpin, LnMV, round(math.log(BPR), 5)])
            y.append(monthly_return - rf)

        monthly_betas = LinearRegression().fit(X, y).coef_.tolist() 
        print(f'complete regression for year {year + 1} month {month}')
        betas.append(monthly_betas)

    return betas

output_file = 'GPIN_Fama_Macbeth_coef.txt'
with open(output_file, 'w') as file:
    for year in range(2002, 2024):
        betas = cross_section_regression(year)
        for beta in betas:
            line = ' '.join(f'{b:.5f}' for b in beta)
            file.write(line + '\n')
        
        betas = cross_section_regression_next_year(year)
        for beta in betas:
            line = ' '.join(f'{b:.5f}' for b in beta)
            file.write(line + '\n')