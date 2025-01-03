import numpy as np
from .utility import normal_density

class ProbMat:
    def __init__(self, mu, sigma):
        (self.mu0, self.mu1), (self.sigma0, self.sigma1) = mu, sigma
        self.loglik = np.zeros([6, 6])
    
    def reset(self):
        self.loglik = np.zeros([6, 6])
    
    def update(self, score, code):
        loglik1 = np.zeros([6, 6])
        loglik1 += normal_density(score,self.mu0,self.sigma0)
        if code <= 6:
            loglik1[code-1, :] = normal_density(score,self.mu1,self.sigma1)
        else:
            loglik1[:, code-7] = normal_density(score,self.mu1,self.sigma1)
        # 
        self.loglik += loglik1
    
    @property
    def probs(self):
        mat = self.loglik-np.max(self.loglik)
        mat = np.exp(mat)
        mat /= np.sum(mat)
        return mat
    
    @property
    def logprobs(self):
        x = self.loglik-np.max(self.loglik)
        dx = np.log(np.sum(np.exp(x)))
        return x-dx
    
    def determine_target(self):
        i = self.probs.argmax()
        return (i//6)+1, (i%6)+7
