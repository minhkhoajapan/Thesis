import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

NUMBER_OF_BETA_PORTFOLIO = 30
STOCK_MISSING_VALUE = -99.9999
MARKET_MISSING_VALUE = -99.99

preranking_beta = pd.DataFrame(np.genfromtxt(
    'preranking_beta.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'beta']
))

preranking_beta['beta_group'] = preranking_beta.groupby('year')['beta'].transform(
    lambda x: pd.qcut(x, q = NUMBER_OF_BETA_PORTFOLIO, labels = list(range(1, NUMBER_OF_BETA_PORTFOLIO + 1, 1)))
)

Mrk_Data = pd.read_excel("risk_free.xlsx", engine="openpyxl", header=None)
Mrk_Data.columns = [
    'year',
    'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
    'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7', 'rf8', 'rf9', 'rf10', 'rf11', 'rf12'
]

SPM_Data = pd.read_excel("SPM.xlsx", engine="openpyxl", header=None)
SPM_Data.columns = ['code', 'year', 'JunPrice', 'LnMV', 'BPR', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']
SPM_Data = SPM_Data.dropna(subset=['LnMV'])

def get_monthly_return(code, year, month):
    row = SPM_Data[(SPM_Data['code'] == code) & (SPM_Data['year'] == year)]
    mrk_row = Mrk_Data[Mrk_Data['year'] == year]

    if row.empty or mrk_row.empty:
        return STOCK_MISSING_VALUE
    
    stock_r = row.iloc[0].get(f'r{month}', np.nan)
    rf = mrk_row.iloc[0].get(f'rf{month}', np.nan)    

    if pd.isna(stock_r) or pd.isna(rf) or stock_r == STOCK_MISSING_VALUE or rf == MARKET_MISSING_VALUE:
        return STOCK_MISSING_VALUE
    
    return stock_r - rf

def get_market_return(year, month):
    mrk_row = Mrk_Data[Mrk_Data['year'] == year]
    if mrk_row.empty:
        return MARKET_MISSING_VALUE
    
    mrk_r = mrk_row.iloc[0].get(f'm{month}', np.nan)
    if pd.isna(mrk_r):
        return MARKET_MISSING_VALUE
    
    return mrk_r

# print(preranking_beta[(preranking_beta['beta_group'] == 1) & (preranking_beta['year'] == 2023)]['code'])


portfolio_beta = []
for portfolio in range(1, NUMBER_OF_BETA_PORTFOLIO + 1, 1):
    X1, X2, y = [], [], []
    for year in range(2002, 2024):
        codes = preranking_beta[(preranking_beta['beta_group'] == portfolio) & (preranking_beta['year'] == year)]['code']
        for month in range(7, 13):
            monthly_portfolio = []
            for code in codes:
                stock_return = get_monthly_return(code, year, month) 
                if stock_return != STOCK_MISSING_VALUE:
                    monthly_portfolio.append(stock_return)
            
            y.append(np.mean(monthly_portfolio))
            X1.append(get_market_return(year, month))
            X2.append(get_market_return(year, month - 1))
        
        for month in range(1, 7):
            monthly_portfolio = []
            for code in codes:
                stock_return = get_monthly_return(code, year + 1, month) 
                if stock_return != STOCK_MISSING_VALUE:
                    monthly_portfolio.append(stock_return)
            
            y.append(np.mean(monthly_portfolio))
            X1.append(get_market_return(year + 1, month))
            if month == 1:
                X2.append(get_market_return(year, 12))
            else:
                X2.append(get_market_return(year + 1, month - 1))
    
    try:
        beta = LinearRegression().fit(np.array(X1).reshape(-1, 1), y).coef_[0]
        lag_beta = LinearRegression().fit(np.array(X2).reshape(-1, 1), y).coef_[0]
        portfolio_beta.append(beta + lag_beta)
    except Exception as e:
        print(f"{e} portfolio {portfolio}")

print(len(portfolio_beta))
output_file = 'portfolio_beta.txt'
with open(output_file, 'w') as file:
    for i in range(len(portfolio_beta)):
        file.write(f"{i + 1} {portfolio_beta[i]:.5f}\n")
