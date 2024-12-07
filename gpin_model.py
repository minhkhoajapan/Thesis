from pandas import Series
from pandas import isnull
import numpy as np
import scipy as sp
from numpy import log, exp, log1p

from scipy.special import gamma, logsumexp
from scipy.linalg import inv
import scipy.optimize as op

from common import *

class GPINModel():
    def __init__(self, a, r, p, eta, th, d, n=1, t=252) -> None:
        r"""
            Initialize parameters of GPIN model

            a: $\alpha$, the unconditional probability of an information event
            r, p: two parameters of Gamma distribution 
            eta: expected number of buys (sells) increases proportionally by eta when there is news arrival
        """
        self.a, self.r, self.p, self.eta, self.th, self.d, self.N, self.T = a, r, p, eta, th, d, n, t

        self.states = self._draw_states()
        self.lamb = _lam(r, p, size=(n, t))
        
        self.buys = np.random.poisson(self.lamb * (th * (self.states != 1) + (th + eta) * (self.states == 1)))
        self.sells = np.random.poisson(self.lamb * ((1 - th) * (self.states != 1) + (1 - th + eta) * (self.states == -1)))

    def _draw_states(self):
        pass


def _lam(r, p, size = None):
    r"""
        Compute $\lambda_{NI} from shape r and scale p/(1-p) params
    """
    return np.nan_to_num(np.random.gamma(r, p/(1-p), size)) 

def _lf(th_b, th_s, r, p, n_buys, n_sells, pdenom=1):
    res =  log(th_b)*n_buys+log(1-th_s)*n_sells - lfact(n_buys) - lfact(n_sells) - gammaln(r) + log(1-p)*r + log(p)*(n_buys+n_sells) + gammaln(r+n_buys+n_sells) - log(pdenom)*r - log(pdenom)*(n_buys+n_sells)
    return res

def _ll(a, r, p, eta, d, th, n_buys, n_sells):
    return np.array([log(1-a)+_lf(th, th, r, p, n_buys, n_sells),
                   log(a*d)+_lf(th+eta, th, r, p, n_buys, n_sells, 1+eta*p),
                   log(a*(1-d))+_lf(th, th-eta, r, p, n_buys, n_sells, 1+eta*p)])

def compute_alpha(a, r, p, eta, d, th, n_buys, n_sells):
    r"""
        Compute the conditional alpha given parameters, buys, and sells.
    """
    ys = _ll(a, r, p, eta, d, th, n_buys, n_sells)

    ymax = ys.max(axis=0)
    lik = exp(ys-ymax)
    alpha = lik[1:].sum(axis=0)/lik.sum(axis=0)

    return alpha

def nbm_ll(theta, x):
    r"""
        Likelihood function of turnover (Buy + Sell)
        utilize in the estimations of alpha, eta, p and r
    """
    a,p,eta,r = theta
    q = (p+eta*p)/(1+eta*p)

    def _nbl(a,p,r,x):
        return log(a)+log(1-p)*r+log(p)*x-lfact(x)-gammaln(r)+gammaln(r+x)
    
    ll = np.array([_nbl(1-a,p,r,x),_nbl(a,q,r,x)])
    return sum(logsumexp(ll,axis=0))

def _loglik(theta, a, r, p, n_buys, n_sells):
    eta,d,th = theta
    
    ll = np.array([log(1-a)+_lf(th, th, r, p, n_buys, n_sells),
                   log(a*d)+_lf(th+eta, th, r, p, n_buys, n_sells, 1+eta*p),
                   log(a*(1-d))+_lf(th, th-eta, r, p, n_buys, n_sells, 1+eta*p)])
    
    return sum(logsumexp(ll,axis=0))

def loglik(theta, n_buys, n_sells):
    a,p,eta,r,d,th = theta
    
    ll = np.array([log(1-a)+_lf(th, th, r, p, n_buys, n_sells),
                   log(a*d)+_lf(th+eta, th, r, p, n_buys, n_sells, 1+eta*p),
                   log(a*(1-d))+_lf(th, th-eta, r, p, n_buys, n_sells, 1+eta*p)])
    
    return sum(logsumexp(ll,axis=0))

def fit(n_buys, n_sells, starts=10, maxiter=100, 
        a=None, r=None, p=None, eta=None, th=None, d=None, 
        se=None, winsorize_turn=False, **kwargs):
    import pandas as pd
    from statsmodels.regression.linear_model import OLS
    
    turn = n_buys + n_sells
    if winsorize_turn:
        sp.stats.mstats.winsorize(turn,limits=0.05,inplace=True)
    ETA_MAX = (abs(n_buys-n_sells)/(n_buys+n_sells)).max()
    
    # estimate negative binomial parameters first
    # a p eta r d th
    nll = lambda *args: -loglik(*args)
    bounds = [(0.00001,0.99999)]*2+[(0.00001,ETA_MAX)]+[(0.00001,np.inf)]+[(0.00001,0.99999)]*2
    ranges = [(0.00001,0.99999)]*2+[(0.00001,ETA_MAX)]+[(0.00001,999)]+[(0.00001,0.99999)]*2

    a0 = a or 0.5
    eta0 = eta or (abs(n_buys-n_sells)/(n_buys+n_sells)).mean()
    d0 = d or 0.5
    p0 = p or (1-(n_buys+n_sells).mean()/(n_buys+n_sells).var())
    r0 = r or (1-p0)/p0*(n_buys+n_sells).mean()
    
    results = OLS(n_sells,n_buys).fit()
    th0 = th or (1/(1+results.params[0]))
        
    res_final = [a0,p0,eta0,r0,d0,th0]
    stderr1,stderr2 = np.zeros(4),np.zeros(2)
    
    nll = lambda *args: -nbm_ll(*args)
    f = nll([a0,p0,eta0,r0],n_buys+n_sells)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            # if any missing or not first iteration try random starts
            if (None in (a,p,eta,r)) or i:
                a,p,eta,r = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges[:4]]
            res = op.minimize(nll, [a0,p0,eta0,r0], method=None,
                              bounds=bounds[:4], args=(turn))
            rc = res['status']
            check_bounds = list(map(lambda x,y: x in y, res['x'], bounds[:4]))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            _,rc = res['fun'],res['status']
            a0,p0,eta0,r0 = res['x']
            stderr1 = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    
    nll = lambda *args: -loglik(*args)
    f = nll([a0,p0,eta0,r0,d0,th0],n_buys,n_sells)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            # if any missing or not first iteration try random starts
            if (None in (a0,p0,eta0,r0,d0,th0)) or i:
                a0,p0,eta0,r0,d0,th0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
            res = op.minimize(nll, [a0,p0,eta0,r0,d0,th0], method=None,
                              bounds=bounds, args=(n_buys,n_sells))
            rc = res['status']
            check_bounds = list(map(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            f,rc = res['fun'],res['status']
            a0,p0,eta0,r0,d0,th0 = res['x']
            stderr2 = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    
    nll = lambda *args: -nbm_ll(*args)
    f = nll([a0,p0,eta0,r0],turn)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            # if any missing or not first iteration try random starts
            if (None in (a,p,eta,r)) or i:
                a,p,eta,r = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges[:4]]
            res = op.minimize(nll, [a0,p0,eta0,r0], method=None,
                              bounds=bounds[:4], args=(turn))
            rc = res['status']
            check_bounds = list(map(lambda x,y: x in y, res['x'], bounds[:4]))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            _,rc = res['fun'],res['status']
            a0,p0,eta0,r0 = res['x']
            stderr1 = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())

    res_final = [a0,p0,eta0,r0,d0,th0]
    param_names = 'a,p,eta,r,d,th'.split(',')
    output = dict(zip(param_names+['f','rc'],
                    res_final+[-loglik(res_final,n_buys,n_sells),rc]))
    if se:
        stderr = stderr1.tolist()+stderr2.tolist()
        output = {'params': dict(zip(param_names,res_final)),
                  'se': dict(zip(param_names,stderr)),
                  'stats':{'f': f,'rc': rc}
                 } 
    return output