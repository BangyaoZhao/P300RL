import numpy as np
import json, random
from collections import deque
from matplotlib.axes import Axes
from .. import utility
from tqdm import tqdm
#
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "basis.json")
with open(DATA_PATH, 'r') as json_file:
    basis_info = json.load(json_file)

def sample_normal(Omega, mu):
    """
    Sample beta from N(Omega^-1 * mu, Omega^-1)

    Parameters:
    Omega (ndarray): A positive definite matrix.
    mu (ndarray): Mean vector.

    Returns:
    beta (ndarray): Sampled vector from N(Omega^-1 * mu, Omega^-1).
    """
    # Step 1: Perform Cholesky decomposition of Omega
    R = np.linalg.cholesky(Omega).T  # R is upper triangular such that Omega = R^T * R
    # Step 2: Solve for b in the equation R^T * b = mu
    b = np.linalg.solve(R.T, mu)
    # Step 3: Sample z ~ N(0, I), where I is the identity matrix of appropriate size
    z = np.random.normal(0, 1, size=mu.shape)
    # Step 4: Solve for beta in the equation R * beta = z + b
    beta = np.linalg.solve(R, z + b)
    return beta

class LinearRegGibbs:
    def __init__(self, X, y, maxlen=500):
        self.L = X.shape[1]
        self.X, self.y = X, y
        self.XtX = X.T @ X
        self.Xty = X.T @ y
        # initialize
        self.initial_params()
        self.recent_params = deque(maxlen=maxlen)
        self.recent_params.append(self.params)
    
    def initial_params(self):
        self.theta = np.random.normal(size=self.L)
        self.sigma2 = 1
        self.tau2 = 1
    
    @property
    def params(self):
        return {'theta': self.theta.copy(), 'sigma2': self.sigma2, 'tau2': self.tau2}
    
    def update_theta(self):
        Omega = self.XtX/self.sigma2 + np.eye(self.L)/self.tau2
        mu = self.Xty/self.sigma2
        self.theta = sample_normal(Omega, mu)
    
    def update_sigma2(self):
        yhat = np.array(self.X) @ self.theta
        scale = 0.1 + 0.5*np.sum(np.square(np.array(self.y) - yhat))
        self.sigma2 = scale/np.random.gamma(0.1+0.5*len(self.y))
    
    def update_tau2(self):
        scale = 0.1 + 0.5*np.sum(np.square(self.theta))
        self.tau2 = scale/np.random.gamma(0.1+0.5*self.L)
    
    def update(self, n_update=1):
        for _ in tqdm(range(n_update)):
            self.update_theta()
            self.update_sigma2()
            self.update_tau2()
            self.recent_params.append(self.params)
    
    def predict(self, X_new):
        k = X_new.shape[0]
        params = random.choices(self.recent_params, k=k)
        Theta = np.stack([param['theta'] for param in params])
        sigma2s = np.array([param['sigma2'] for param in params])
        return np.sum(X_new*Theta, axis=1) + np.sqrt(sigma2s)*np.random.normal(size=k) 

class GP:
    def __init__(self, x, y) -> None:
        self.x_grid, self.Psi, self.eigvals = np.array(basis_info['x']), np.array(basis_info['Psi']), np.array(basis_info['lambda'])
        self.x_grid = self.x_grid.flatten()
        self.n_basis, self.n_time = self.Psi.shape
        #
        self.x, self.y = x, y
        self.sampler = LinearRegGibbs(self.get_newX(x), y)
    
    def get_newX(self, newx):
        locs = np.argmin(np.abs(self.x_grid.reshape([1, -1]) - newx.reshape([-1, 1])), axis=1)
        return self.Psi[locs, :] * np.sqrt(self.eigvals).reshape([1, -1])
    
    def update(self, n_update=1):
        self.sampler.update(n_update)
        
    def predict(self, x_new):
        return self.sampler.predict(self.get_newX(x_new))
    
    def visualize(self, ax: Axes):
        gp_noise_samples = np.stack([self.predict(self.x_grid) for _ in range(500)])
        ####################
        lb, ub = np.quantile(gp_noise_samples, [0.025, 0.975], axis=0)
        ax.plot(self.x_grid, lb, color='skyblue')
        ax.plot(self.x_grid, ub, color='skyblue')
        ax.fill_between(self.x_grid, lb, ub, color='skyblue', alpha=0.2)
        #
        ax.scatter(self.x, self.y, color='red', s=1)
        ax.set_xlim(0, 1)
    
    def simulate(self, logpmat: np.ndarray, actions):
        logpmat = logpmat.copy()
        if not actions:
            return logpmat
        else:
            action = actions[0]
            if action <= 5:
                logp_a = utility.log_sum_exp(logpmat[action])
            else:
                logp_a = utility.log_sum_exp(logpmat[:, action-6])
            logodds_a = utility.logit_exp(logp_a)
            logoddsratio = self.predict(np.exp([logp_a]))[0]
            logodds_a_new = logodds_a + logoddsratio
            dlogp_a = utility.log_sigmoid(logodds_a_new) - utility.log_sigmoid(logodds_a)
            dlogp_na = utility.log_sigmoid(-logodds_a_new) - utility.log_sigmoid(-logodds_a)
            #
            logpmat += dlogp_na
            if action <= 5:
                logpmat[action] += dlogp_a - dlogp_na
            else:
                logpmat[:, action-6] += dlogp_a - dlogp_na
            return self.simulate(logpmat, actions[1:])
            
            
        