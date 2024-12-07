import numpy as np
from numpy import log, exp

from scipy.special import logsumexp
from scipy.linalg import inv
import scipy.optimize as op

from common import *

class EOModel():
    def __init__(self, a, d, es, eb, u, n = 1, t=252) -> None:
        r"""
            Initialize parameters for Easley and O'Hara Sequential Trade Model

            a: $\alpha$, the unconditional probability of an information event
            d: $\delta$, the unconditional probability of good news
            es: $\epsilon_s$, the average number of sells on a day with no news
            eb: $\epsilon_b$, the average number of buys on a day with no news
            u: $\mu$, the average number of (additional) trades on a day with news
            
            n: the number of stocks to simulate, default 1
            t: the number of periods to simulate, default 252(one trading year)
        """
        self.a, self.d, self.es, self.eb, self.u, self.N, self.T = a, d, es, eb, u, n, t

def _lf(eb, es, n_buys, n_sells):
    r"""
        Log Likelihood of trades
    
        Likelihood is derived from the Poisson distribution 
    """
    return -eb + n_buys*log(eb) - lfact(n_buys) - es + n_sells*log(es) - lfact(n_sells)

def _ll(a, d, eb, es, u, n_buys, n_sells):
    r"""
        extends _lf to incorporate the Easley-O'Hara model states:
          - Good news: log(a * d)
          - Bad news: log(a * (1-d))
          - No news: log(1 - a)
    """
    return np.array([log(a * (1-d)) + _lf(eb, es + u, n_buys, n_sells),
                    log(a * d) + _lf(eb + u, es, n_buys, n_sells), 
                    log(1-a) + _lf(eb, es, n_buys, n_sells)])

def loglik(theta, n_buys, n_sells):
    a, d, eb, es, u = theta
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)
    
    return sum(logsumexp(ll, axis=0))

def fit(n_buys, n_sells, starts = 10, maxiter = 100, a = None, d = None, eb = None, es = None, u = None, se = None, **kwargs):
    nll = lambda *args: -loglik(*args)

    bounds = [(0.00001, 0.99999)] * 2 + [(0.00001, np.inf)] * 3
    ranges = [(0.00001, 0.99999)] * 2

    a0, d0 = [x or 0.5 for x in (a, d)]
    eb0, es0 = eb or np.mean(n_buys), es or np.mean(n_sells)
    oib = n_buys - n_sells
    u0 = u or np.mean(abs(oib))

    res_final = [a0, d0, eb0, es0, u0]
    stderr = np.zeros_like(res_final)
    f = nll(res_final, n_buys, n_sells)

    for i in range(starts):
        rc = -1
        j = 0
        while rc != 0 and j <= maxiter:
            if (None in (res_final)) or i:
                a0, d0 = [np.random.uniform(l, np.nan_to_num(h)) for (l, h) in ranges]
                eb0, es0, u0 = np.random.poisson([eb, es, u])

            res = op.minimize(nll, [a0, d0, eb0, es0, u0], bounds=bounds, args=(n_buys, n_sells))
            rc = res['status']

            check_bounds = list(map(lambda x, y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            
            j += 1
        
        if res['success'] and res['fun'] <= f:
            f, rc = res['fun'], res['status']
            res_final = res['x'].tolist()
            stderr = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
        
        param_names = ['a', 'd', 'eb', 'es', 'u']
        output = dict(zip(param_names + ['f', 'rc'], res_final + [f, rc]))

        #if user require stats on standard errors
        if se:
            output = {
                'params': dict(zip(param_names, res_final)),
                'se': dict(zip(param_names, stderr)),
                'stats': {'f': f, 'rc': rc}
            }
        
        return output


