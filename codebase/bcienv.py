import gymnasium as gym 
import numpy as np
import random
from .probmats import ProbMat

class SimulatedEnv(gym.Env):
    def __init__(self, probmat: ProbMat, 
                 t_chr = 3.5/60, t_flash = (31.25+125)/1000/60,
                 score_mu = [0.1, 0.5], score_sigma = [0.2, 0.2], p0 = 0.5):
        self.probmat, self.t_chr, self.t_flash = probmat, t_chr, t_flash
        self.score_mu, self.score_sigma, self.p0 = score_mu, score_sigma, p0
        # process data
        self.current_chr_id = -1
        self.time = 0
    
    def _get_score(self, ft):
        if self.is_previous_target and random.random()<self.p0:
            ft_effective = 0
        else:
            ft_effective = ft
        self.is_previous_target = (ft==1)
        return np.random.normal(self.score_mu[ft_effective], self.score_sigma[ft_effective])

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # 
        self._move_to_next_chr()
        #
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _move_to_next_chr(self):
        self.current_chr_id += 1
        self.current_flash_id = 0
        self.time += self.t_chr
        self.current_target = (random.randint(0, 5)+1, random.randint(0, 5)+7)
        # reset mat and add true target
        self.is_previous_target = False
        self.probmat.reset()

    def step(self, action):
        assert action in list(range(13))
        if action == 12:
            if self.probmat.determine_target()==self.current_target:
                reward = np.log(36-1)/np.log(2)
            else:
                reward = -np.log(36-1)/np.log(2)
            self._move_to_next_chr()
        else:
            reward = 0
            ft = int(action+1 in self.current_target)
            self.probmat.update(self._get_score(ft), action+1)
            self.current_flash_id += 1
            self.time += self.t_flash
        return self._get_obs(), reward, False, False, self._get_info()

    def _get_obs(self):
        obs = {
            'certainty_scores': self.probmat.probs,
            'log_certainty_scores': self.probmat.logprobs,
            'time': self.time,
            'current_chr_id': self.current_chr_id,
            'current_flash_id': self.current_flash_id,
            'correct_if_stop_now': self.probmat.determine_target()==self.current_target
        }
        return obs

    def _get_info(self):
        return None
    
    def close(self):
        pass   
