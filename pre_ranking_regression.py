import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


SPM_Data = pd.read_excel("SPM.xlsx", engine="openpyxl", header=None)
SPM_Data.columns = ['code', 'year', 'JunPrice', 'LnMV', 'BPR', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']
SPM_Data = SPM_Data.dropna(subset=['LnMV'])

Mrk_Data = pd.read_excel("risk_free.xlsx", engine="openpyxl", header=None)
Mrk_Data.columns = [
    'year',
    'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
    'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7', 'rf8', 'rf9', 'rf10', 'rf11', 'rf12'
]

def preranking_beta_monthly_data(code, end_year, min_months = 24, max_months = 60, stock_missing_value = -99.99999, market_missing_value = -99.99, last_available_year = 2000):
    monthly_returns, market_excess_return, lag_market_excess_return, risk_free = [], [], [], []
    month_count = 0
    current_year = end_year
    month_indices = list(range(6, 0, -1))

    while current_year >= last_available_year:
        row = SPM_Data[(SPM_Data['code'] == code) & (SPM_Data['year'] == current_year)]
        mkr_row = Mrk_Data[Mrk_Data['year'] == current_year]

        if row.empty or mkr_row.empty:
            current_year -= 1
            month_indices = list(range(12, 0, -1))
            continue

        for i in month_indices:

            stock_r = row.iloc[0].get(f'r{i}', np.nan)
            market_excess_r = mkr_row.iloc[0].get(f'm{i}', np.nan)
            rf = mkr_row.iloc[0].get(f'rf{i}', np.nan)
            lag_market_excess_r = market_missing_value
            if i == 1:
                lag_mkr_row = Mrk_Data[Mrk_Data['year'] == current_year - 1]
                lag_market_excess_r = lag_mkr_row.iloc[0].get(f'm{12}', np.nan)
            else:
                lag_market_excess_r = mkr_row.iloc[0].get(f'm{i-1}', np.nan)

            if any(pd.isna(v) for v in [stock_r,market_excess_r, rf, lag_market_excess_r]):
                continue

            if stock_r == stock_missing_value or market_excess_r == -99.99 or rf == market_missing_value or lag_market_excess_r == market_missing_value:
                continue

            monthly_returns.append(stock_r)
            market_excess_return.append(market_excess_r)
            risk_free.append(rf)
            lag_market_excess_return.append(lag_market_excess_r)
            month_count += 1

            if month_count >= max_months:
                break
        
        if month_count >= max_months:
            break
            
        current_year -= 1
        month_indices = list(range(12, 0, -1))
    
    if len(monthly_returns) < min_months:
        return None, None, None
    
    monthly_excess_returns = np.array(monthly_returns) - np.array(risk_free)
    return monthly_excess_returns, np.array(market_excess_return), np.array(lag_market_excess_return)

# stock_excess, market_excess, lag_market_excess = preranking_beta_monthly_data(2483, 2008)
# beta = LinearRegression().fit(market_excess.reshape(-1,1), stock_excess).coef_[0]
# lag_beta = LinearRegression().fit(lag_market_excess.reshape(-1, 1), stock_excess).coef_[0]
# print(f"{beta:.5f} {lag_beta:.5f} {beta+lag_beta:.5f}")

all_codes = SPM_Data['code'].unique().tolist()
output_file = 'preranking_beta.txt'

with open(output_file, 'w') as file:
    for code in all_codes:
        for year in range(2002, 2024):
            stock_excess, market_excess, lag_market_excess = preranking_beta_monthly_data(code, year)   
            if stock_excess is None or len(stock_excess) < 24:
                continue
            try:
                beta = LinearRegression().fit(market_excess.reshape(-1, 1), stock_excess).coef_[0]
                lag_beta = LinearRegression().fit(lag_market_excess.reshape(-1, 1), stock_excess).coef_[0]
                file.write(f"{code} {year} {beta+lag_beta:.5f}\n")
                print(f"beta of {code} in year {year} written to {output_file}")
            except Exception as e:
                print(f"{e} code {code} in year {year}")

