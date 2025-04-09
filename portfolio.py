import numpy as np
import pandas as pd

SPM_Data = pd.read_excel("SPM.xlsx", engine="openpyxl", header=None)
SPM_Data.columns = ['code', 'year', 'JunPrice', 'LnMV', 'BPR', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']
SPM_Data = SPM_Data.dropna(subset=['LnMV'])

PIN_txt = np.genfromtxt(
    'PIN_results.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'pin_a', 'pin_d', 'pin_eb', 'pin_es', 'pin_u' ,'pin_f', 'pin_rc', 'PIN']
)
PIN_Data = pd.DataFrame(PIN_txt)

# Mrk_Data = pd.read_excel("risk_free.xlsx", engine="openpyxl", header=None)
# Mrk_Data.columns = [
#     'year',
#     'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
#     'rf1', 'rf2', 'rf3', 'rf4', 'rf5', 'rf6', 'rf7', 'rf8', 'rf9', 'rf10', 'rf11', 'rf12'
# ]

shift_cols = ['pin_a', 'pin_d', 'pin_eb', 'pin_es', 'pin_u' ,'pin_f', 'pin_rc', 'PIN']
PIN_Data[shift_cols] = PIN_Data.groupby('code')[shift_cols].shift(1)

PIN_Data = PIN_Data.dropna(subset=shift_cols)

SPM_Data['yearly_return'] = SPM_Data.groupby("code")["JunPrice"].pct_change(fill_method=None)
SPM_Data['yearly_return'] = SPM_Data.groupby('code')['yearly_return'].shift(-1)
SPM_Data['MV'] = np.exp(SPM_Data['LnMV'])
SPM_Data['yearly_return'] *= 100

SPM_Data = SPM_Data.merge(PIN_Data, on = ['code', 'year'], how='inner')
# SPM_Data["MV_group"] = SPM_Data.groupby("year")["LnMV"].transform(
#     lambda x: pd.qcut(x.rank(method="first"), q=3, labels=["Small", "Mid", "Large"])
# )
SPM_Data['MV_group'] = SPM_Data.groupby('year')['MV'].transform(lambda x: pd.qcut(x, q = 5, labels=['Small', '2', '3', '4', 'Large'], duplicates = 'drop'))
SPM_Data['PIN_group'] = SPM_Data.groupby('year')['PIN'].transform(lambda x: pd.qcut(x, q = 3, labels=['Low', 'Medium', 'High'], duplicates= 'drop'))


grouped_returns = (
    SPM_Data.groupby(["year", "MV_group", "PIN_group"]).agg(
        yearly_return = ('yearly_return', 'mean'),
        MV = ('MV', 'mean'),
        code = ('code', 'count'),
        PIN = ('PIN', 'mean')
    ).reset_index()
)

final_mean_returns = (
    grouped_returns.groupby(["MV_group", "PIN_group"]).agg(
        yearly_return = ('yearly_return', 'mean'),
        MV = ('MV', 'mean'),
        code = ('code', 'mean'),
        PIN = ('PIN', 'mean')
    ).reset_index()
)

pannel_A = final_mean_returns.pivot(index = 'MV_group', columns = 'PIN_group', values = 'yearly_return')
print('\nPannel A: Return')
print(pannel_A)

pannel_B = final_mean_returns.pivot(index = 'MV_group', columns = 'PIN_group', values = 'MV')
print('\nPannel A: Market Value')
print(pannel_B)

pannel_C = final_mean_returns.pivot(index = 'MV_group', columns = 'PIN_group', values = 'code')
print('\nPannel C: Number of Stocks')
print(pannel_C)

pannel_D = final_mean_returns.pivot(index = 'MV_group', columns = 'PIN_group', values = 'PIN')
print('\nPannel D: PIN')
print(pannel_D)
