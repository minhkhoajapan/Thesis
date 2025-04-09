import numpy as np
import pandas as pd

coefs = pd.DataFrame(np.genfromtxt(
    'PIN_Fama_Macbeth_coef.txt',
    delimiter = ' ',
    dtype = None,
    encoding = 'utf-8',
    names = ['beta_coef', 'PIN_coef', 'LnMV_coef', 'BPR_coef']
))

beta = coefs['beta_coef']
PIN = coefs['PIN_coef']
LnMV = coefs['LnMV_coef']
BPR = coefs['BPR_coef']

print(f"beta: {beta.mean():.5f} PIN: {PIN.mean():.5f} LnMV: {LnMV.mean():.5f} BPR: {BPR.mean():.5f}")
beta_standard_error = np.std(beta, ddof=1) / np.sqrt(len(beta))
PIN_standard_error = np.std(PIN, ddof=1) / np.sqrt(len(PIN))
LnMV_standard_error = np.std(LnMV, ddof=1) / np.sqrt(len(LnMV))
BPR_standard_error = np.std(BPR, ddof=1) / np.sqrt(len(BPR))

beta_t = np.mean(beta) / beta_standard_error
PIN_t = np.mean(PIN) / PIN_standard_error
LnMV_t = np.mean(LnMV) / LnMV_standard_error
BPR_t = np.mean(BPR) / BPR_standard_error
print(f"beta_t: {beta_t:.5f} PIN_t: {PIN_t:.5f} LnMV_t: {LnMV_t:.5f} BPR_t: {BPR_t:.5f}")