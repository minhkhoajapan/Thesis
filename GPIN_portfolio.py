import numpy as np
import pandas as pd

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

# Mrk_Data = pd.read_excel("risk_free.xlsx", engine="openpyxl", header=None)
# Mrk_Data.columns = [
#     'year',
#     'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
#     'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7', 'rf8', 'rf9', 'rf10', 'rf11', 'rf12'
# ]

shift_cols = ['gpin_a', 'gpin_p', 'gpin_eta', 'gpin_r', 'gpin_d', 'gpin_th', 'gpin_f', 'gpin_rc', 'GPIN']
GPIN_Data[shift_cols] = GPIN_Data.groupby('code')[shift_cols].shift(1)

GPIN_Data = GPIN_Data.dropna(subset=shift_cols)

SPM_Data['yearly_return'] = SPM_Data.groupby("code")["JunPrice"].pct_change(fill_method=None)
SPM_Data['yearly_return'] = SPM_Data.groupby('code')['yearly_return'].shift(-1)
SPM_Data['MV'] = np.exp(SPM_Data['LnMV'])
SPM_Data['yearly_return'] *= 100

SPM_Data = SPM_Data.merge(GPIN_Data, on = ['code', 'year'], how='inner')
SPM_Data["MV_group"] = SPM_Data.groupby("year")["LnMV"].transform(
    lambda x: pd.qcut(x.rank(method="first"), q=3, labels=["Small", "Mid", "Large"])
)
SPM_Data['GPIN_group'] = SPM_Data.groupby('year')['GPIN'].transform(lambda x: pd.qcut(x, q = 3, labels=['Low', 'Medium', 'High'], duplicates= 'drop'))


grouped_returns = (
    SPM_Data.groupby(["year", "MV_group", "GPIN_group"]).agg(
        yearly_return = ('yearly_return', 'mean'),
        MV = ('MV', 'mean'),
        code = ('code', 'count'),
        GPIN = ('GPIN', 'mean')
    ).reset_index()
)

final_mean_returns = (
    grouped_returns.groupby(["MV_group", "GPIN_group"]).agg(
        yearly_return = ('yearly_return', 'mean'),
        MV = ('MV', 'mean'),
        code = ('code', 'mean'),
        GPIN = ('GPIN', 'mean')
    ).reset_index()
)

pannel_A = final_mean_returns.pivot(index = 'MV_group', columns = 'GPIN_group', values = 'yearly_return')
print('\nPannel A: Return')
print(pannel_A)

pannel_B = final_mean_returns.pivot(index = 'MV_group', columns = 'GPIN_group', values = 'MV')
print('\nPannel A: Market Value')
print(pannel_B)

pannel_C = final_mean_returns.pivot(index = 'MV_group', columns = 'GPIN_group', values = 'code')
print('\nPannel C: Number of Stocks')
print(pannel_C)

pannel_D = final_mean_returns.pivot(index = 'MV_group', columns = 'GPIN_group', values = 'GPIN')
print('\nPannel D: GPIN')
print(pannel_D)