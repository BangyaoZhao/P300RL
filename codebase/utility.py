import numpy as np

def cal_utility(n_correct, n_wrong, t):
    n_correct = n_correct - n_wrong
    return np.log(36-1)/np.log(2)*n_correct/t

def cal_t(n_chr, n_flash, t_chr = 3.5/60, t_flash = (31.25+125)/1000/60):
    return n_chr*t_chr+n_flash*t_flash

def normal_density(x, mu=0, sigma=1):
    return -np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * ((x - mu) / sigma) ** 2

def logit_exp(logp):
    return logp - np.log1p(-np.exp(logp))

def log_sigmoid(logodds):
    if logodds<=0:
        return logodds-np.log1p(np.exp(logodds))
    else:
        return -np.log1p(np.exp(-logodds))

def log_sum_exp(logps):
    x_max = np.max(logps)
    x = logps - x_max
    dx = np.log(np.sum(np.exp(x)))
    return x_max + dx